"""
classical_models.py - Classical ML classifier pipelines with hyperparameter tuning.

Implements:
- Logistic Regression
- SVM (RBF and linear kernels)
- Random Forest
- Gradient Boosting (sklearn)
- XGBoost (if available)
- LightGBM (if available)

All models are wrapped in a unified ModelWrapper with:
- GridSearchCV / RandomizedSearchCV
- Class weight / imbalance handling
- Probability calibration
- Feature importance extraction
"""

from __future__ import annotations

import logging
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Unified model wrapper
# ---------------------------------------------------------------------------

class ModelWrapper:
    """Wraps a sklearn-compatible classifier with auto-tuning and calibration.

    Args:
        name: Human-readable model name.
        estimator: sklearn estimator instance.
        param_grid: Dict of hyperparameter lists for grid/random search.
        cv_folds: Cross-validation folds for hyperparameter search.
        n_iter: Iterations for RandomizedSearchCV (None = full grid).
        calibrate: Whether to apply probability calibration post-fit.
        seed: Random seed.
    """

    def __init__(
        self,
        name: str,
        estimator: BaseEstimator,
        param_grid: Dict[str, List[Any]],
        cv_folds: int = 5,
        n_iter: Optional[int] = 50,
        calibrate: bool = False,
        seed: int = 42,
    ) -> None:
        self.name = name
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.calibrate = calibrate
        self.seed = seed
        self._fitted_model: Optional[BaseEstimator] = None
        self._best_params: Dict[str, Any] = {}
        self._train_time: float = 0.0

    @property
    def is_fitted(self) -> bool:
        return self._fitted_model is not None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ModelWrapper":
        """Fit model with hyperparameter search.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Optional validation features (unused for classical ML).
            y_val: Optional validation labels (unused for classical ML).

        Returns:
            Self.
        """
        n_classes = len(np.unique(y_train))
        min_class = min(np.unique(y_train, return_counts=True)[1])

        # Adjust CV folds if dataset is tiny
        effective_folds = min(self.cv_folds, min_class)
        if effective_folds < 2:
            logger.warning(
                "%s: only %d samples in smallest class. Fitting without CV.",
                self.name, min_class,
            )
            t0 = time.perf_counter()
            self.estimator.fit(X_train, y_train)
            self._fitted_model = self.estimator
            self._train_time = time.perf_counter() - t0
            return self

        cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=self.seed)

        # Determine whether to use Randomized or Grid search
        n_candidates = 1
        for vals in self.param_grid.values():
            n_candidates *= len(vals)

        if self.n_iter is not None and n_candidates > self.n_iter:
            logger.info(
                "%s: RandomizedSearchCV (%d candidates → %d iterations)",
                self.name, n_candidates, self.n_iter,
            )
            search = RandomizedSearchCV(
                self.estimator,
                self.param_grid,
                n_iter=self.n_iter,
                cv=cv,
                scoring="f1_weighted",
                n_jobs=-1,
                random_state=self.seed,
                verbose=0,
                refit=True,
            )
        else:
            from sklearn.model_selection import GridSearchCV
            logger.info(
                "%s: GridSearchCV (%d candidates)", self.name, n_candidates,
            )
            search = GridSearchCV(
                self.estimator,
                self.param_grid,
                cv=cv,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=0,
                refit=True,
            )

        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        self._train_time = time.perf_counter() - t0

        self._best_params = search.best_params_
        best_estimator = search.best_estimator_

        if self.calibrate:
            logger.info("%s: Applying probability calibration (sigmoid).", self.name)
            cal_cv = StratifiedKFold(
                n_splits=min(3, effective_folds), shuffle=True, random_state=self.seed
            )
            calibrated = CalibratedClassifierCV(
                best_estimator, cv=cal_cv, method="sigmoid"
            )
            calibrated.fit(X_train, y_train)
            self._fitted_model = calibrated
        else:
            self._fitted_model = best_estimator

        logger.info(
            "%s trained in %.1fs | best params: %s",
            self.name, self._train_time, self._best_params,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            1D integer array of predicted labels.
        """
        self._check_fitted()
        return self._fitted_model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            2D array of shape (n_samples, n_classes).
        """
        self._check_fitted()
        if hasattr(self._fitted_model, "predict_proba"):
            return self._fitted_model.predict_proba(X)  # type: ignore[union-attr]
        # Fallback: decision_function → sigmoid
        scores = self._fitted_model.decision_function(X)  # type: ignore[union-attr]
        proba = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - proba, proba])

    def feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances if available.

        Returns:
            1D array of importances, or None.
        """
        model = self._fitted_model
        # Unwrap calibration
        if hasattr(model, "base_estimator"):
            model = model.base_estimator
        if hasattr(model, "estimator"):
            model = model.estimator
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        if hasattr(model, "coef_"):
            coef = model.coef_
            return np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        return None

    def save(self, path: Union[str, Path]) -> None:
        """Save fitted model to disk.

        Args:
            path: File path (pickle format).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Saved %s to %s", self.name, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelWrapper":
        """Load a fitted ModelWrapper from disk.

        Args:
            path: Pickle file path.

        Returns:
            Loaded ModelWrapper.
        """
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("Loaded model from %s", path)
        return obj

    def _check_fitted(self) -> None:
        if self._fitted_model is None:
            raise RuntimeError(f"{self.name} has not been fitted. Call fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"ModelWrapper(name={self.name!r}, status={status})"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_logistic_regression(config: Dict[str, Any], seed: int = 42) -> ModelWrapper:
    """Build Logistic Regression wrapper from config.

    Args:
        config: Full project config dict.
        seed: Random seed.

    Returns:
        Un-fitted :class:`ModelWrapper`.
    """
    cfg = config.get("classical_models", {}).get("logistic_regression", {})
    estimator = LogisticRegression(
        class_weight=cfg.get("class_weight", "balanced"),
        max_iter=cfg.get("max_iter", 1000),
        random_state=seed,
        solver="saga",
        multi_class="auto",
    )
    param_grid = {
        "C": cfg.get("C", [0.001, 0.01, 0.1, 1.0, 10.0]),
        "penalty": cfg.get("penalty", ["l2"]),
    }
    return ModelWrapper(
        name="LogisticRegression",
        estimator=estimator,
        param_grid=param_grid,
        cv_folds=cfg.get("cv_folds", 5),
        seed=seed,
    )


def build_svm(config: Dict[str, Any], seed: int = 42) -> ModelWrapper:
    """Build SVM classifier wrapper from config.

    Args:
        config: Full project config dict.
        seed: Random seed.

    Returns:
        Un-fitted :class:`ModelWrapper`.
    """
    cfg = config.get("classical_models", {}).get("svm", {})
    estimator = SVC(
        probability=True,
        class_weight=cfg.get("class_weight", "balanced"),
        random_state=seed,
    )
    param_grid = {
        "C": cfg.get("C", [0.1, 1.0, 10.0]),
        "kernel": cfg.get("kernel", ["rbf", "linear"]),
        "gamma": cfg.get("gamma", ["scale", "auto"]),
    }
    return ModelWrapper(
        name="SVM",
        estimator=estimator,
        param_grid=param_grid,
        cv_folds=cfg.get("cv_folds", 5),
        seed=seed,
    )


def build_random_forest(config: Dict[str, Any], seed: int = 42) -> ModelWrapper:
    """Build Random Forest wrapper from config.

    Args:
        config: Full project config dict.
        seed: Random seed.

    Returns:
        Un-fitted :class:`ModelWrapper`.
    """
    cfg = config.get("classical_models", {}).get("random_forest", {})
    estimator = RandomForestClassifier(
        class_weight=cfg.get("class_weight", "balanced"),
        random_state=seed,
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators": cfg.get("n_estimators", [100, 200]),
        "max_depth": cfg.get("max_depth", [None, 10, 20]),
        "min_samples_split": cfg.get("min_samples_split", [2, 5]),
    }
    return ModelWrapper(
        name="RandomForest",
        estimator=estimator,
        param_grid=param_grid,
        cv_folds=cfg.get("cv_folds", 5),
        seed=seed,
    )


def build_gradient_boosting(config: Dict[str, Any], seed: int = 42) -> ModelWrapper:
    """Build sklearn GradientBoosting wrapper from config.

    Args:
        config: Full project config dict.
        seed: Random seed.

    Returns:
        Un-fitted :class:`ModelWrapper`.
    """
    cfg = config.get("classical_models", {}).get("gradient_boosting", {})
    estimator = GradientBoostingClassifier(random_state=seed)
    param_grid = {
        "n_estimators": cfg.get("n_estimators", [100, 200]),
        "learning_rate": cfg.get("learning_rate", [0.05, 0.1]),
        "max_depth": cfg.get("max_depth", [3, 5]),
        "subsample": cfg.get("subsample", [0.8, 1.0]),
    }
    return ModelWrapper(
        name="GradientBoosting",
        estimator=estimator,
        param_grid=param_grid,
        cv_folds=cfg.get("cv_folds", 5),
        seed=seed,
    )


def build_xgboost(
    config: Dict[str, Any],
    seed: int = 42,
    class_ratio: float = 1.0,
) -> Optional[ModelWrapper]:
    """Build XGBoost wrapper from config (if xgboost is installed).

    Args:
        config: Full project config dict.
        seed: Random seed.
        class_ratio: Positive class weight ratio (n_neg / n_pos).

    Returns:
        Un-fitted :class:`ModelWrapper`, or None if xgboost not installed.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed; skipping XGBoost.")
        return None

    cfg = config.get("classical_models", {}).get("xgboost", {})
    scale_pos = cfg.get("scale_pos_weight", None) or class_ratio

    estimator = XGBClassifier(
        scale_pos_weight=scale_pos,
        random_state=seed,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators": cfg.get("n_estimators", [100, 200]),
        "learning_rate": cfg.get("learning_rate", [0.05, 0.1]),
        "max_depth": cfg.get("max_depth", [3, 5]),
        "subsample": cfg.get("subsample", [0.8]),
    }
    return ModelWrapper(
        name="XGBoost",
        estimator=estimator,
        param_grid=param_grid,
        cv_folds=cfg.get("cv_folds", 5),
        seed=seed,
    )


def build_lightgbm(config: Dict[str, Any], seed: int = 42) -> Optional[ModelWrapper]:
    """Build LightGBM wrapper from config (if lightgbm is installed).

    Args:
        config: Full project config dict.
        seed: Random seed.

    Returns:
        Un-fitted :class:`ModelWrapper`, or None if lightgbm not installed.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        logger.warning("lightgbm not installed; skipping LightGBM.")
        return None

    cfg = config.get("classical_models", {}).get("lightgbm", {})
    estimator = LGBMClassifier(
        class_weight=cfg.get("class_weight", "balanced"),
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    param_grid = {
        "n_estimators": cfg.get("n_estimators", [100, 200]),
        "learning_rate": cfg.get("learning_rate", [0.05, 0.1]),
        "num_leaves": cfg.get("num_leaves", [31, 63]),
    }
    return ModelWrapper(
        name="LightGBM",
        estimator=estimator,
        param_grid=param_grid,
        cv_folds=cfg.get("cv_folds", 5),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_all_classical_models(
    config: Dict[str, Any],
    seed: int = 42,
    class_counts: Optional[Dict[int, int]] = None,
) -> List[ModelWrapper]:
    """Build all enabled classical model wrappers.

    Args:
        config: Full project config dict.
        seed: Random seed.
        class_counts: Optional dict {label: count} for computing class ratio.

    Returns:
        List of un-fitted :class:`ModelWrapper` objects.
    """
    classic_cfg = config.get("classical_models", {})
    models: List[ModelWrapper] = []

    if classic_cfg.get("logistic_regression", {}).get("enabled", True):
        models.append(build_logistic_regression(config, seed))

    if classic_cfg.get("svm", {}).get("enabled", True):
        models.append(build_svm(config, seed))

    if classic_cfg.get("random_forest", {}).get("enabled", True):
        models.append(build_random_forest(config, seed))

    if classic_cfg.get("gradient_boosting", {}).get("enabled", True):
        models.append(build_gradient_boosting(config, seed))

    class_ratio = 1.0
    if class_counts and len(class_counts) == 2:
        counts = list(class_counts.values())
        class_ratio = max(counts) / max(min(counts), 1)

    if classic_cfg.get("xgboost", {}).get("enabled", True):
        m = build_xgboost(config, seed, class_ratio)
        if m is not None:
            models.append(m)

    if classic_cfg.get("lightgbm", {}).get("enabled", True):
        m = build_lightgbm(config, seed)
        if m is not None:
            models.append(m)

    logger.info(
        "Built %d classical model wrappers: %s",
        len(models),
        [m.name for m in models],
    )
    return models
