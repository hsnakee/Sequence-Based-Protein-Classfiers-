"""
ensemble.py - Ensemble and stacking classifiers combining multiple models.

Implements:
- Soft / hard voting ensemble
- Stacking with configurable meta-learner
- Probability averaging
- Per-model weight support
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from utils import get_logger

logger = get_logger(__name__)


class VotingEnsemble:
    """Soft or hard voting ensemble over multiple base classifiers.

    Each base classifier must expose a ``predict_proba(X)`` method
    (for soft voting) or ``predict(X)`` (for hard voting).

    Args:
        models: List of fitted classifier objects (ModelWrapper or NeuralModelTrainer).
        strategy: "soft" for probability averaging; "hard" for majority vote.
        weights: Optional 1D array of per-model weights (must sum > 0).
        class_labels: Mapping from int label to class name.
    """

    def __init__(
        self,
        models: List[Any],
        strategy: str = "soft",
        weights: Optional[np.ndarray] = None,
        class_labels: Optional[Dict[int, str]] = None,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required for VotingEnsemble.")
        self.models = models
        self.strategy = strategy.lower()
        self.weights = weights
        self.class_labels = class_labels or {}
        self._n_classes: int = 2

    @property
    def name(self) -> str:
        return f"VotingEnsemble_{self.strategy}"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities by averaging member probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Averaged probability matrix of shape (n_samples, n_classes).
        """
        all_probas = []
        for model in self.models:
            try:
                prob = model.predict_proba(X)
                all_probas.append(prob)
            except Exception as e:
                logger.warning("Model %s failed predict_proba: %s", getattr(model, "name", model), e)

        if not all_probas:
            raise RuntimeError("All models failed predict_proba.")

        probas = np.array(all_probas)  # (n_models, n_samples, n_classes)

        if self.weights is not None:
            w = np.array(self.weights[: len(all_probas)], dtype=float)
            w = w / w.sum()
            averaged = np.average(probas, axis=0, weights=w)
        else:
            averaged = probas.mean(axis=0)

        return averaged

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        For soft voting: argmax of averaged probabilities.
        For hard voting: majority vote over member predictions.

        Args:
            X: Feature matrix.

        Returns:
            1D integer label array.
        """
        if self.strategy == "soft":
            return self.predict_proba(X).argmax(axis=1)

        # Hard voting: collect discrete predictions
        all_preds = []
        for model in self.models:
            try:
                all_preds.append(model.predict(X))
            except Exception as e:
                logger.warning("Model %s failed predict: %s", getattr(model, "name", model), e)

        if not all_preds:
            raise RuntimeError("All models failed predict.")

        votes = np.array(all_preds)  # (n_models, n_samples)
        # Majority vote per sample
        from scipy.stats import mode
        result, _ = mode(votes, axis=0, keepdims=False)
        return result.astype(int)

    def save(self, path: Union[str, Path]) -> None:
        """Save ensemble to disk.

        Args:
            path: Output pickle path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Saved VotingEnsemble to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VotingEnsemble":
        """Load ensemble from disk.

        Args:
            path: Pickle file path.

        Returns:
            Loaded VotingEnsemble.
        """
        with open(path, "rb") as fh:
            return pickle.load(fh)


class StackingEnsemble:
    """Stacking classifier: base models → meta-learner.

    Trains base models using out-of-fold predictions to avoid leakage,
    then trains a meta-learner on stacked probability features.

    Args:
        base_models: List of (name, unfitted_model_factory_fn) tuples.
            Each factory must return a model with fit/predict_proba.
        meta_learner: sklearn-compatible classifier for the meta layer.
        cv_folds: Folds for generating out-of-fold predictions.
        seed: Random seed.
    """

    def __init__(
        self,
        meta_learner: Optional[Any] = None,
        cv_folds: int = 5,
        seed: int = 42,
    ) -> None:
        self.meta_learner = meta_learner or LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        )
        self.cv_folds = cv_folds
        self.seed = seed
        self._fitted_base_models: List[Any] = []
        self._fitted = False

    @property
    def name(self) -> str:
        return "StackingEnsemble"

    def fit(
        self,
        base_models: List[Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "StackingEnsemble":
        """Fit stacking ensemble.

        Args:
            base_models: List of already-fitted base model objects.
            X_train: Training feature matrix.
            y_train: Training labels.

        Returns:
            Self.
        """
        logger.info("Fitting StackingEnsemble with %d base models.", len(base_models))
        self._fitted_base_models = base_models
        n = len(y_train)

        # Generate out-of-fold predictions from base models
        skf = StratifiedKFold(
            n_splits=min(self.cv_folds, min(np.unique(y_train, return_counts=True)[1])),
            shuffle=True, random_state=self.seed,
        )

        meta_features = np.zeros((n, len(base_models)), dtype=np.float32)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]

            for m_idx, model in enumerate(base_models):
                import copy
                m_copy = copy.deepcopy(model)
                try:
                    m_copy.fit(X_tr, y_tr)
                    oof_proba = m_copy.predict_proba(X_va)
                    meta_features[val_idx, m_idx] = oof_proba[:, 1]
                except Exception as e:
                    logger.warning(
                        "Stacking fold %d model %s failed: %s",
                        fold_idx, getattr(model, "name", m_idx), e,
                    )

        self.meta_learner.fit(meta_features, y_train)
        self._fitted = True
        logger.info("StackingEnsemble meta-learner fitted.")
        return self

    def _build_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Stack base model probabilities into meta-feature matrix."""
        cols = []
        for model in self._fitted_base_models:
            try:
                proba = model.predict_proba(X)[:, 1]
            except Exception:
                proba = np.zeros(len(X))
            cols.append(proba)
        return np.column_stack(cols)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via meta-learner.

        Args:
            X: Feature matrix.

        Returns:
            2D probability array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        meta_X = self._build_meta_features(X)
        return self.meta_learner.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            1D integer label array.
        """
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: Union[str, Path]) -> None:
        """Save to disk.

        Args:
            path: Output pickle path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Saved StackingEnsemble to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "StackingEnsemble":
        """Load from disk.

        Args:
            path: Pickle file path.

        Returns:
            Loaded StackingEnsemble.
        """
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ensembles(
    config: Dict[str, Any],
    classical_models: List[Any],
    neural_models: List[Any],
) -> List[Any]:
    """Build voting and stacking ensembles from trained model lists.

    Args:
        config: Full project config dict.
        classical_models: List of fitted classical ModelWrapper objects.
        neural_models: List of fitted NeuralModelTrainer objects.

    Returns:
        List of ensemble objects (VotingEnsemble and/or StackingEnsemble).
    """
    ensemble_cfg = config.get("ensemble", {})
    if not ensemble_cfg.get("enabled", True):
        return []

    all_models = classical_models + neural_models
    if len(all_models) < 2:
        logger.warning("Need ≥2 models for ensemble; skipping.")
        return []

    ensembles: List[Any] = []
    methods = ensemble_cfg.get("methods", ["voting"])
    strategy = ensemble_cfg.get("voting_strategy", "soft")

    if "voting" in methods:
        ve = VotingEnsemble(all_models, strategy=strategy)
        ensembles.append(ve)
        logger.info("Built VotingEnsemble (%s) with %d models.", strategy, len(all_models))

    if "stacking" in methods:
        se = StackingEnsemble(cv_folds=ensemble_cfg.get("cv_folds", 5))
        ensembles.append(se)
        logger.info("Built StackingEnsemble (unfitted; call fit() explicitly).")

    return ensembles
