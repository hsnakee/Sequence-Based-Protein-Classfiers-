"""
embedding_models.py - Transformer-based protein embeddings with disk caching.

Supports:
- ProtBERT (Rostlab/prot_bert_bfd)
- ESM-2 (esm2_t33_650M_UR50D and variants)

Features:
- Automatic GPU/CPU detection
- Batch inference with progress bars
- Disk caching keyed on sequence content hash
- Memory-safe processing (processes in batches)
- Graceful degradation when transformers not installed
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from utils import get_device, get_logger, ensure_dir, md5_of_strings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Abstract base for sequence embedding models.

    Subclasses must implement :meth:`_embed_batch` and :attr:`embedding_dim`.
    Disk caching and batching are handled in the base class.
    """

    def __init__(
        self,
        batch_size: int = 8,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
        device: str = "auto",
    ) -> None:
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        self._device = get_device(device)
        self._model_loaded = False

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""
        ...

    @abstractmethod
    def _load_model(self) -> None:
        """Load model weights (called lazily on first embed call)."""
        ...

    @abstractmethod
    def _embed_batch(self, sequences: List[str]) -> np.ndarray:
        """Embed a batch of sequences.

        Args:
            sequences: List of amino acid strings (same length for efficiency).

        Returns:
            Array of shape (len(sequences), embedding_dim).
        """
        ...

    def _cache_key(self, sequence: str) -> str:
        return hashlib.md5(sequence.encode()).hexdigest()

    def _load_from_cache(self, key: str) -> Optional[np.ndarray]:
        if not self.use_cache or self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            try:
                return np.load(path)
            except Exception as e:
                logger.warning("Cache load failed for %s: %s", key, e)
        return None

    def _save_to_cache(self, key: str, vec: np.ndarray) -> None:
        if not self.use_cache or self.cache_dir is None:
            return
        ensure_dir(self.cache_dir)
        path = self.cache_dir / f"{key}.npy"
        try:
            np.save(path, vec)
        except Exception as e:
            logger.warning("Cache save failed for %s: %s", key, e)

    def embed(self, sequences: List[str], show_progress: bool = True) -> np.ndarray:
        """Embed a list of sequences with caching and batching.

        Args:
            sequences: List of amino acid strings.
            show_progress: Show tqdm progress bar.

        Returns:
            Array of shape (len(sequences), embedding_dim).
        """
        if not self._model_loaded:
            logger.info("Loading %s model...", self.__class__.__name__)
            self._load_model()
            self._model_loaded = True

        # Separate cached from uncached
        results: Dict[int, np.ndarray] = {}
        uncached_indices: List[int] = []
        uncached_seqs: List[str] = []

        for i, seq in enumerate(sequences):
            key = self._cache_key(seq)
            cached = self._load_from_cache(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_seqs.append(seq)

        logger.info(
            "%s: %d cached, %d to embed",
            self.__class__.__name__,
            len(results),
            len(uncached_seqs),
        )

        # Batch process uncached sequences
        if uncached_seqs:
            batch_iterator = range(0, len(uncached_seqs), self.batch_size)
            if show_progress:
                batch_iterator = tqdm(
                    batch_iterator,
                    desc=f"{self.__class__.__name__} embedding",
                    unit="batch",
                )

            for batch_start in batch_iterator:
                batch_seqs = uncached_seqs[batch_start : batch_start + self.batch_size]
                try:
                    batch_embeddings = self._embed_batch(batch_seqs)
                except Exception as e:
                    logger.error("Embedding batch failed: %s. Using zeros.", e)
                    batch_embeddings = np.zeros(
                        (len(batch_seqs), self.embedding_dim), dtype=np.float32
                    )

                for j, (seq, emb) in enumerate(zip(batch_seqs, batch_embeddings)):
                    idx = uncached_indices[batch_start + j]
                    results[idx] = emb
                    self._save_to_cache(self._cache_key(seq), emb)

        # Assemble in order
        output = np.vstack([results[i] for i in range(len(sequences))])
        return output.astype(np.float32)


# ---------------------------------------------------------------------------
# ProtBERT
# ---------------------------------------------------------------------------

class ProtBERTEmbedder(BaseEmbedder):
    """Embed protein sequences using ProtBERT (Rostlab/prot_bert_bfd).

    Sequences must have spaces between amino acids (ProtBERT convention).
    Outputs mean-pooled last hidden state.

    Args:
        model_name: HuggingFace model identifier.
        max_length: Maximum tokenised length (ProtBERT max = 512 tokens).
        **kwargs: Forwarded to :class:`BaseEmbedder`.
    """

    def __init__(
        self,
        model_name: str = "Rostlab/prot_bert_bfd",
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None

    @property
    def embedding_dim(self) -> int:
        return 1024  # ProtBERT hidden size

    def _load_model(self) -> None:
        try:
            from transformers import BertModel, BertTokenizer
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for ProtBERT embeddings. "
                "Install with: pip install transformers torch"
            ) from e

        logger.info("Loading ProtBERT: %s", self.model_name)
        self._tokenizer = BertTokenizer.from_pretrained(
            self.model_name, do_lower_case=False
        )
        self._model = BertModel.from_pretrained(self.model_name)
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("ProtBERT loaded on device: %s", self._device)

    def _format_sequence(self, seq: str) -> str:
        """Insert spaces between amino acids for ProtBERT tokeniser."""
        return " ".join(list(seq))

    def _embed_batch(self, sequences: List[str]) -> np.ndarray:
        import torch

        formatted = [self._format_sequence(s) for s in sequences]
        encoding = self._tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)
            hidden_states = outputs.last_hidden_state  # (B, L, 1024)

            # Mean pool (excluding padding tokens)
            attention_mask = encoding["attention_mask"].unsqueeze(-1)
            masked = hidden_states * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            embeddings = (summed / counts).cpu().numpy()

        return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# ESM-2
# ---------------------------------------------------------------------------

class ESMEmbedder(BaseEmbedder):
    """Embed protein sequences using Meta's ESM-2.

    Uses the ``esm`` library (fair-esm). Falls back gracefully if not installed.

    Args:
        model_name: ESM model identifier string.
        max_length: Maximum sequence length before truncation.
        **kwargs: Forwarded to :class:`BaseEmbedder`.
    """

    ESM_MODELS = {
        "esm2_t6_8M_UR50D": (6, 320),
        "esm2_t12_35M_UR50D": (12, 480),
        "esm2_t30_150M_UR50D": (30, 640),
        "esm2_t33_650M_UR50D": (33, 1280),
        "esm2_t36_3B_UR50D": (36, 2560),
    }

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        max_length: int = 1022,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_length = max_length
        self._model: Any = None
        self._alphabet: Any = None
        self._batch_converter: Any = None
        _, self._emb_dim = self.ESM_MODELS.get(model_name, (33, 1280))

    @property
    def embedding_dim(self) -> int:
        return self._emb_dim

    def _load_model(self) -> None:
        try:
            import esm
            import torch
        except ImportError as e:
            raise ImportError(
                "fair-esm is required for ESM embeddings. "
                "Install with: pip install fair-esm"
            ) from e

        logger.info("Loading ESM model: %s", self.model_name)
        model_fn = getattr(esm.pretrained, self.model_name, None)
        if model_fn is None:
            raise ValueError(f"Unknown ESM model: {self.model_name}")

        self._model, self._alphabet = model_fn()
        self._batch_converter = self._alphabet.get_batch_converter()
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("ESM model loaded on device: %s", self._device)

    def _embed_batch(self, sequences: List[str]) -> np.ndarray:
        import torch

        # Truncate to ESM max length
        truncated = [(f"seq_{i}", s[: self.max_length]) for i, s in enumerate(sequences)]
        _, _, tokens = self._batch_converter(truncated)
        tokens = tokens.to(self._device)

        # Get num layers for representation extraction
        n_layers, _ = self.ESM_MODELS.get(self.model_name, (33, 1280))

        with torch.no_grad():
            results = self._model(tokens, repr_layers=[n_layers], return_contacts=False)
        token_reps = results["representations"][n_layers]  # (B, L+2, dim)

        # Mean pool (excluding BOS/EOS tokens)
        embeddings = token_reps[:, 1:-1, :].mean(dim=1).cpu().numpy()
        return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_EMBEDDER_REGISTRY: Dict[str, type] = {
    "protbert": ProtBERTEmbedder,
    "esm": ESMEmbedder,
}


def get_embedder(name: str, config: Dict[str, Any]) -> BaseEmbedder:
    """Instantiate an embedder by name from the project config.

    Args:
        name: Embedder name; one of "protbert" or "esm".
        config: Full project config dict.

    Returns:
        Instantiated :class:`BaseEmbedder` subclass.

    Raises:
        ValueError: If ``name`` is not a registered embedder.
    """
    if name not in _EMBEDDER_REGISTRY:
        raise ValueError(
            f"Unknown embedder '{name}'. Available: {list(_EMBEDDER_REGISTRY.keys())}"
        )

    emb_cfg = config.get("embeddings", {}).get(name, {})
    cls = _EMBEDDER_REGISTRY[name]

    common_kwargs: Dict[str, Any] = {
        "batch_size": emb_cfg.get("batch_size", 8),
        "cache_dir": emb_cfg.get("cache_dir", f".embedding_cache/{name}"),
        "use_cache": emb_cfg.get("use_cache", True),
        "device": emb_cfg.get("device", "auto"),
    }

    if name == "protbert":
        return cls(
            model_name=emb_cfg.get("model_name", "Rostlab/prot_bert_bfd"),
            max_length=emb_cfg.get("max_length", 512),
            **common_kwargs,
        )
    elif name == "esm":
        return cls(
            model_name=emb_cfg.get("model_name", "esm2_t33_650M_UR50D"),
            max_length=emb_cfg.get("max_length", 1022),
            **common_kwargs,
        )

    return cls(**common_kwargs)


def list_available_embedders() -> List[str]:
    """Return list of registered embedder names."""
    return list(_EMBEDDER_REGISTRY.keys())


def clear_embedding_cache(cache_dir: Union[str, Path]) -> int:
    """Delete all cached embedding files.

    Args:
        cache_dir: Directory containing cached .npy files.

    Returns:
        Number of files deleted.
    """
    cache_dir = Path(cache_dir)
    count = 0
    if cache_dir.exists():
        for f in cache_dir.glob("*.npy"):
            f.unlink()
            count += 1
    logger.info("Cleared %d cached embeddings from %s", count, cache_dir)
    return count
