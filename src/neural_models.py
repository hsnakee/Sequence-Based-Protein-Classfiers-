"""
neural_models.py - PyTorch neural network classifiers for protein sequences.

Implements:
- CNN on one-hot encoded sequences
- BiLSTM/GRU model on one-hot encoded sequences  
- Positional Transformer encoder + classifier head

All models share:
- Configurable architecture from the project config
- Early stopping
- Dropout regularisation
- Class-weighted loss
- GPU/CPU auto-detection
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from utils import get_device, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy PyTorch import helper
# ---------------------------------------------------------------------------

def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for neural models. "
            "Install with: pip install torch"
        ) from e


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor validation loss and stop training when it stops improving.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum improvement to count as progress.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss: float = float("inf")
        self._wait: int = 0

    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._wait = 0
        else:
            self._wait += 1
        return self._wait >= self.patience


# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------

def build_cnn_model(
    seq_length: int,
    n_classes: int = 2,
    filters: List[int] = (64, 128, 256),
    kernel_sizes: List[int] = (3, 5, 7),
    pool_size: int = 2,
    dense_units: List[int] = (256, 64),
    dropout: float = 0.3,
) -> "torch.nn.Module":  # type: ignore[name-defined]
    """Build a 1D CNN classifier for protein sequences.

    Architecture:
        Input (one-hot) → Conv1D blocks with MaxPool → GlobalAvgPool
        → Dense layers → Sigmoid/Softmax output

    Args:
        seq_length: Input sequence length (padded/truncated).
        n_classes: Number of output classes.
        filters: Number of filters per conv layer.
        kernel_sizes: Kernel size per conv layer.
        pool_size: MaxPool kernel size.
        dense_units: Neurons per dense layer.
        dropout: Dropout probability.

    Returns:
        A ``torch.nn.Module``.
    """
    _require_torch()
    import torch
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            conv_layers: List[nn.Module] = []
            in_channels = 20  # one-hot = 20 amino acids
            for n_filt, ks in zip(filters, kernel_sizes):
                conv_layers += [
                    nn.Conv1d(in_channels, n_filt, kernel_size=ks, padding=ks // 2),
                    nn.BatchNorm1d(n_filt),
                    nn.ReLU(),
                    nn.MaxPool1d(pool_size),
                    nn.Dropout(dropout),
                ]
                in_channels = n_filt
            self.conv_stack = nn.Sequential(*conv_layers)
            self.global_pool = nn.AdaptiveAvgPool1d(1)

            dense_layers: List[nn.Module] = []
            in_dim = in_channels
            for out_dim in dense_units:
                dense_layers += [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_dim = out_dim
            dense_layers.append(nn.Linear(in_dim, n_classes))
            self.dense_stack = nn.Sequential(*dense_layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, seq_length, 20) → transpose to (B, 20, seq_length)
            x = x.permute(0, 2, 1)
            x = self.conv_stack(x)           # (B, last_filters, L')
            x = self.global_pool(x).squeeze(-1)  # (B, last_filters)
            return self.dense_stack(x)       # (B, n_classes)

    return CNN()


# ---------------------------------------------------------------------------
# BiLSTM Model
# ---------------------------------------------------------------------------

def build_bilstm_model(
    n_classes: int = 2,
    hidden_size: int = 128,
    num_layers: int = 2,
    bidirectional: bool = True,
    dense_units: List[int] = (128, 64),
    dropout: float = 0.3,
) -> "torch.nn.Module":  # type: ignore[name-defined]
    """Build a Bidirectional LSTM classifier.

    Architecture:
        Embedding (one-hot) → BiLSTM → Last hidden state
        → Dense layers → Output

    Args:
        n_classes: Number of output classes.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        bidirectional: Use bidirectional LSTM.
        dense_units: Dense layer sizes.
        dropout: Dropout probability.

    Returns:
        A ``torch.nn.Module``.
    """
    _require_torch()
    import torch
    import torch.nn as nn

    class BiLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=20,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            directions = 2 if bidirectional else 1
            lstm_out_dim = hidden_size * directions

            dense_layers: List[nn.Module] = [nn.Dropout(dropout)]
            in_dim = lstm_out_dim
            for out_dim in dense_units:
                dense_layers += [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_dim = out_dim
            dense_layers.append(nn.Linear(in_dim, n_classes))
            self.dense_stack = nn.Sequential(*dense_layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, seq_len, 20)
            _, (h_n, _) = self.lstm(x)   # h_n: (num_layers * dirs, B, H)
            # Concatenate last forward and backward hidden states
            if self.lstm.bidirectional:
                h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2H)
            else:
                h = h_n[-1]  # (B, H)
            return self.dense_stack(h)

    return BiLSTM()


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------

def build_transformer_model(
    seq_length: int,
    n_classes: int = 2,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dense_units: List[int] = (128, 64),
    dropout: float = 0.3,
) -> "torch.nn.Module":  # type: ignore[name-defined]
    """Build a Transformer encoder classifier.

    Architecture:
        Amino acid linear projection → Positional encoding
        → Transformer encoder layers → Mean pool → Dense → Output

    Args:
        seq_length: Maximum sequence length.
        n_classes: Number of output classes.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_layers: Number of encoder layers.
        dim_feedforward: Feedforward network dimension.
        dense_units: Dense layer sizes.
        dropout: Dropout probability.

    Returns:
        A ``torch.nn.Module``.
    """
    _require_torch()
    import math
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer("pe", pe)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.pe[:, : x.size(1), :]  # type: ignore[index]
            return self.dropout(x)

    class TransformerClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(20, d_model)
            self.pos_enc = PositionalEncoding(d_model, max_len=seq_length + 10, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            dense_layers: List[nn.Module] = []
            in_dim = d_model
            for out_dim in dense_units:
                dense_layers += [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_dim = out_dim
            dense_layers.append(nn.Linear(in_dim, n_classes))
            self.classifier = nn.Sequential(*dense_layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, seq_len, 20)
            x = self.input_proj(x)       # (B, seq_len, d_model)
            x = self.pos_enc(x)
            x = self.transformer(x)      # (B, seq_len, d_model)
            x = x.mean(dim=1)            # Global mean pool
            return self.classifier(x)

    return TransformerClassifier()


# ---------------------------------------------------------------------------
# Neural model trainer
# ---------------------------------------------------------------------------

class NeuralModelTrainer:
    """Wraps a PyTorch model with training, validation, and prediction logic.

    Args:
        model: PyTorch nn.Module.
        name: Human-readable name.
        config: Full project config dict.
        device: torch.device.
        class_weights: Optional tensor of class weights for loss.
    """

    def __init__(
        self,
        model: "torch.nn.Module",  # type: ignore[name-defined]
        name: str,
        config: Dict[str, Any],
        device: Optional["torch.device"] = None,  # type: ignore[name-defined]
        class_weights: Optional["torch.Tensor"] = None,  # type: ignore[name-defined]
    ) -> None:
        _require_torch()
        import torch

        self.name = name
        self.config = config
        self._device = device or get_device(config.get("neural_models", {}).get("device", "auto"))
        self.model = model.to(self._device)
        self._class_weights = class_weights
        self._train_history: List[Dict[str, float]] = []
        self._fitted = False

        neural_cfg = config.get("neural_models", {})
        self.epochs = neural_cfg.get("epochs", 50)
        self.patience = neural_cfg.get("patience", 10)
        self.batch_size = neural_cfg.get("batch_size", 32)
        self.lr = neural_cfg.get("learning_rate", 0.001)
        self.weight_decay = neural_cfg.get("weight_decay", 1e-4)
        self.grad_clip = neural_cfg.get("gradient_clip", 1.0)

    def _make_dataloader(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        shuffle: bool = True,
    ) -> "torch.utils.data.DataLoader":  # type: ignore[name-defined]
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_t = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y_t = torch.tensor(y, dtype=torch.long)
            ds = TensorDataset(X_t, y_t)
        else:
            ds = TensorDataset(X_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "NeuralModelTrainer":
        """Train the neural model with early stopping.

        Args:
            X_train: Training features (N, seq_len, 20) or (N, seq_len*20).
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).

        Returns:
            Self.
        """
        import torch
        import torch.nn as nn

        # Reshape if flat one-hot
        if X_train.ndim == 2:
            seq_len = X_train.shape[1] // 20
            X_train = X_train.reshape(-1, seq_len, 20)
            if X_val is not None:
                X_val = X_val.reshape(-1, seq_len, 20)

        train_loader = self._make_dataloader(X_train, y_train, shuffle=True)
        val_loader = (
            self._make_dataloader(X_val, y_val, shuffle=False)
            if X_val is not None and y_val is not None
            else None
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.patience // 2, factor=0.5, verbose=False
        )
        loss_fn = nn.CrossEntropyLoss(
            weight=self._class_weights.to(self._device) if self._class_weights is not None else None
        )
        early_stop = EarlyStopping(patience=self.patience)

        t0 = time.perf_counter()
        best_val_loss = float("inf")
        best_state: Optional[Dict] = None

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                X_b, y_b = batch[0].to(self._device), batch[1].to(self._device)
                optimizer.zero_grad()
                logits = self.model(X_b)
                loss = loss_fn(logits, y_b)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validate
            val_loss = train_loss
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        X_b, y_b = batch[0].to(self._device), batch[1].to(self._device)
                        logits = self.model(X_b)
                        val_loss += loss_fn(logits, y_b).item()
                val_loss /= len(val_loader)

            scheduler.step(val_loss)
            self._train_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if epoch % 10 == 0:
                logger.debug(
                    "%s epoch %d/%d — train_loss=%.4f val_loss=%.4f",
                    self.name, epoch, self.epochs, train_loss, val_loss,
                )

            if early_stop(val_loss):
                logger.info("%s: early stopping at epoch %d", self.name, epoch)
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._fitted = True
        logger.info(
            "%s trained in %.1fs | best_val_loss=%.4f",
            self.name, time.perf_counter() - t0, best_val_loss,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature array.

        Returns:
            2D array of shape (n_samples, n_classes).
        """
        import torch
        import torch.nn.functional as F

        if X.ndim == 2:
            seq_len = X.shape[1] // 20
            X = X.reshape(-1, seq_len, 20)

        loader = self._make_dataloader(X, y=None, shuffle=False)
        self.model.eval()
        probs = []

        with torch.no_grad():
            for (X_b,) in loader:
                X_b = X_b.to(self._device)
                logits = self.model(X_b)
                probs.append(F.softmax(logits, dim=-1).cpu().numpy())

        return np.vstack(probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature array.

        Returns:
            1D integer array.
        """
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def save(self, path: Union[str, Path]) -> None:
        """Save the trainer (model + config) to disk.

        Args:
            path: Output file path (.pt or .pkl).
        """
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "name": self.name,
            "history": self._train_history,
        }, path)
        logger.info("Saved %s to %s", self.name, path)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        n_params = sum(p.numel() for p in self.model.parameters())
        return f"NeuralModelTrainer(name={self.name!r}, params={n_params:,}, status={status})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_all_neural_models(
    config: Dict[str, Any],
    seq_length: int,
    n_classes: int = 2,
    class_weights: Optional[np.ndarray] = None,
) -> List[NeuralModelTrainer]:
    """Build all enabled neural models.

    Args:
        config: Full project config dict.
        seq_length: Input sequence length for models.
        n_classes: Number of output classes.
        class_weights: Optional 1D array of per-class weights.

    Returns:
        List of :class:`NeuralModelTrainer` objects (unfitted).
    """
    _require_torch()
    import torch

    neural_cfg = config.get("neural_models", {})
    device = get_device(neural_cfg.get("device", "auto"))
    dropout = neural_cfg.get("dropout", 0.3)

    cw_tensor = None
    if class_weights is not None:
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32)

    trainers: List[NeuralModelTrainer] = []

    # CNN
    if neural_cfg.get("cnn", {}).get("enabled", True):
        cnn_cfg = neural_cfg.get("cnn", {})
        model = build_cnn_model(
            seq_length=seq_length,
            n_classes=n_classes,
            filters=cnn_cfg.get("filters", [64, 128, 256]),
            kernel_sizes=cnn_cfg.get("kernel_sizes", [3, 5, 7]),
            pool_size=cnn_cfg.get("pool_size", 2),
            dense_units=cnn_cfg.get("dense_units", [256, 64]),
            dropout=dropout,
        )
        trainers.append(NeuralModelTrainer(model, "CNN", config, device, cw_tensor))

    # BiLSTM
    if neural_cfg.get("bilstm", {}).get("enabled", True):
        lstm_cfg = neural_cfg.get("bilstm", {})
        model = build_bilstm_model(
            n_classes=n_classes,
            hidden_size=lstm_cfg.get("hidden_size", 128),
            num_layers=lstm_cfg.get("num_layers", 2),
            bidirectional=lstm_cfg.get("bidirectional", True),
            dense_units=lstm_cfg.get("dense_units", [128, 64]),
            dropout=dropout,
        )
        trainers.append(NeuralModelTrainer(model, "BiLSTM", config, device, cw_tensor))

    # Transformer
    if neural_cfg.get("transformer", {}).get("enabled", True):
        tf_cfg = neural_cfg.get("transformer", {})
        model = build_transformer_model(
            seq_length=seq_length,
            n_classes=n_classes,
            d_model=tf_cfg.get("d_model", 128),
            nhead=tf_cfg.get("nhead", 4),
            num_layers=tf_cfg.get("num_layers", 2),
            dim_feedforward=tf_cfg.get("dim_feedforward", 256),
            dense_units=tf_cfg.get("dense_units", [128, 64]),
            dropout=dropout,
        )
        trainers.append(NeuralModelTrainer(model, "Transformer", config, device, cw_tensor))

    logger.info(
        "Built %d neural model trainers: %s",
        len(trainers),
        [t.name for t in trainers],
    )
    return trainers
