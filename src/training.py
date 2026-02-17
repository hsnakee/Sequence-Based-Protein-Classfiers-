"""
training.py - Unified training orchestrator for all model families.

Coordinates:
- Dataset loading and splitting
- Feature extraction
- Classical ML model training with GridSearchCV
- Neural model training with early stopping
- Cross-validation evaluation
- Result aggregation and artifact saving
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from data_loading import ProteinDataset, DataSplitter
from feature_extractors import FeaturePipeline, build_feature_pipeline, build_onehot_pipeline
from classical_models import ModelWrapper, build_all_classical_models
from neural_models import NeuralModelTrainer, build_all_neural_models
from evaluation import ModelEvaluator, compute_metrics, find_optimal_threshold
from utils import get_logger, ensure_dir, save_json, Timer, print_banner, format_metrics_table

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Artifact bundle
# ---------------------------------------------------------------------------

class TrainingArtifacts:
    """Container for all objects produced during training.

    Attributes:
        feature_pipeline: Fitted classical-ML feature pipeline.
        onehot_pipeline: Fitted one-hot pipeline (for neural models).
        classical_models: List of fitted ModelWrapper objects.
        neural_models: List of fitted NeuralModelTrainer objects.
        evaluator: Populated ModelEvaluator.
        config: Project configuration dict used for training.
        class_labels: Mapping from int label to class name.
    """

    def __init__(self) -> None:
        self.feature_pipeline: Optional[FeaturePipeline] = None
        self.onehot_pipeline: Optional[FeaturePipeline] = None
        self.classical_models: List[ModelWrapper] = []
        self.neural_models: List[NeuralModelTrainer] = []
        self.evaluator: Optional[ModelEvaluator] = None
        self.config: Dict[str, Any] = {}
        self.class_labels: Dict[int, str] = {}
        self.best_model_name: str = ""
        self.threshold: float = 0.5

    def save(self, output_dir: Union[str, Path]) -> None:
        """Persist all training artifacts to disk.

        Args:
            output_dir: Directory to write artifacts into.
        """
        output_dir = ensure_dir(output_dir)

        # Feature pipelines
        if self.feature_pipeline is not None:
            with open(output_dir / "feature_pipeline.pkl", "wb") as fh:
                pickle.dump(self.feature_pipeline, fh)

        if self.onehot_pipeline is not None:
            with open(output_dir / "onehot_pipeline.pkl", "wb") as fh:
                pickle.dump(self.onehot_pipeline, fh)

        # Classical models
        for model in self.classical_models:
            model.save(output_dir / f"classical_{model.name}.pkl")

        # Neural models (saved as .pt)
        for trainer in self.neural_models:
            trainer.save(output_dir / f"neural_{trainer.name}.pt")

        # Config and metadata
        save_json(self.config, output_dir / "config.json")
        save_json({
            "class_labels": {str(k): v for k, v in self.class_labels.items()},
            "best_model": self.best_model_name,
            "threshold": self.threshold,
        }, output_dir / "metadata.json")

        logger.info("All training artifacts saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates end-to-end training across all model families.

    Args:
        config: Full project configuration dict.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.seed = config.get("project", {}).get("seed", 42)
        self.output_dir = ensure_dir(config.get("project", {}).get("output_dir", "results"))
        self.artifacts = TrainingArtifacts()
        self.artifacts.config = config

    def run(
        self,
        dataset: ProteinDataset,
        train_classical: bool = True,
        train_neural: bool = True,
    ) -> TrainingArtifacts:
        """Execute the full training pipeline.

        Args:
            dataset: Labelled :class:`ProteinDataset`.
            train_classical: Whether to train classical ML models.
            train_neural: Whether to train neural models.

        Returns:
            Populated :class:`TrainingArtifacts`.
        """
        print_banner("Protein Sequence Classifier — Training")
        self.artifacts.class_labels = dataset.label_names

        # 1. Split dataset
        splitter = DataSplitter(
            test_size=self.config.get("training", {}).get("test_size", 0.2),
            val_size=self.config.get("training", {}).get("val_size", 0.1),
            cv_folds=self.config.get("training", {}).get("cv_folds", 5),
            seed=self.seed,
        )
        train_ds, val_ds, test_ds = splitter.split(dataset)
        logger.info(
            "Split: train=%d | val=%d | test=%d",
            len(train_ds), len(val_ds), len(test_ds),
        )

        # 2. Evaluator
        evaluator = ModelEvaluator(
            output_dir=self.output_dir / "plots",
            label_names=dataset.label_names,
            show_plots=self.config.get("visualization", {}).get("show_plots", False),
            plot_format=self.config.get("visualization", {}).get("plot_format", "png"),
            dpi=self.config.get("visualization", {}).get("dpi", 150),
        )
        self.artifacts.evaluator = evaluator

        # ---- Classical ML ----
        if train_classical:
            with Timer("Classical ML feature extraction"):
                feat_pipeline = build_feature_pipeline(self.config)
                X_train_cl = feat_pipeline.fit_transform(train_ds.sequences)
                X_val_cl = feat_pipeline.transform(val_ds.sequences)
                X_test_cl = feat_pipeline.transform(test_ds.sequences)
                self.artifacts.feature_pipeline = feat_pipeline
                logger.info(
                    "Classical features: %d samples × %d features",
                    X_train_cl.shape[0], X_train_cl.shape[1],
                )

            all_X_train = np.vstack([X_train_cl, X_val_cl])
            all_y_train = np.concatenate([train_ds.labels, val_ds.labels])

            classical_wrappers = build_all_classical_models(
                self.config, seed=self.seed, class_counts=dataset.class_counts
            )

            for wrapper in classical_wrappers:
                if not self.config.get("classical_models", {}).get(
                    wrapper.name.lower().replace(" ", "_"), {}
                ).get("enabled", True):
                    continue

                with Timer(f"Train {wrapper.name}"):
                    try:
                        wrapper.fit(all_X_train, all_y_train)
                    except Exception as e:
                        logger.error("Failed to train %s: %s", wrapper.name, e)
                        continue

                y_pred = wrapper.predict(X_test_cl)
                y_proba = wrapper.predict_proba(X_test_cl)
                feat_imps = wrapper.feature_importances()

                evaluator.evaluate_model(
                    model_name=wrapper.name,
                    y_true=test_ds.labels,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    feature_names=feat_pipeline.feature_names,
                    feature_importances=feat_imps,
                )
                self.artifacts.classical_models.append(wrapper)

        # ---- Neural Models ----
        if train_neural:
            try:
                import torch
            except ImportError:
                logger.warning("PyTorch not installed; skipping neural models.")
                train_neural = False

        if train_neural:
            with Timer("One-hot feature extraction"):
                oh_pipeline = build_onehot_pipeline(self.config)
                X_train_oh = oh_pipeline.fit_transform(train_ds.sequences)
                X_val_oh = oh_pipeline.transform(val_ds.sequences)
                X_test_oh = oh_pipeline.transform(test_ds.sequences)
                self.artifacts.onehot_pipeline = oh_pipeline
                seq_length = self.config.get("features", {}).get("one_hot", {}).get("max_length", 1000)
                logger.info(
                    "One-hot features: %d samples × %d features",
                    X_train_oh.shape[0], X_train_oh.shape[1],
                )

            cw_dict = dataset.class_weights
            cw_array = np.array([cw_dict.get(i, 1.0) for i in sorted(cw_dict)])

            neural_trainers = build_all_neural_models(
                self.config,
                seq_length=seq_length,
                n_classes=dataset.n_classes,
                class_weights=cw_array,
            )

            for trainer in neural_trainers:
                with Timer(f"Train {trainer.name}"):
                    try:
                        trainer.fit(X_train_oh, train_ds.labels, X_val_oh, val_ds.labels)
                    except Exception as e:
                        logger.error("Failed to train %s: %s", trainer.name, e)
                        continue

                y_pred = trainer.predict(X_test_oh)
                y_proba = trainer.predict_proba(X_test_oh)

                evaluator.evaluate_model(
                    model_name=trainer.name,
                    y_true=test_ds.labels,
                    y_pred=y_pred,
                    y_proba=y_proba,
                )
                self.artifacts.neural_models.append(trainer)

        # ---- Comparison table ----
        comparison = evaluator.comparison_table()
        evaluator.save_comparison_table()
        evaluator.plot_model_comparison(metric="f1_weighted")

        if not comparison.empty:
            logger.info(
                "\n%s",
                comparison[["Model", "accuracy", "f1_weighted", "roc_auc", "mcc"]].to_string(index=False),
            )
            best_row = comparison.iloc[0]
            self.artifacts.best_model_name = str(best_row["Model"])
            logger.info("🏆 Best model: %s (F1=%.3f)", self.artifacts.best_model_name, best_row.get("f1_weighted", 0))

        # Save all artifacts
        self.artifacts.save(self.output_dir)
        return self.artifacts


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

def cross_validate_model(
    model_wrapper: ModelWrapper,
    feature_pipeline: FeaturePipeline,
    dataset: ProteinDataset,
    config: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Run stratified k-fold cross-validation for a single classical model.

    Args:
        model_wrapper: Un-fitted ModelWrapper.
        feature_pipeline: Un-fitted FeaturePipeline.
        dataset: Full labelled dataset.
        config: Project config dict.
        seed: Random seed.

    Returns:
        Dict mapping metric name to list of per-fold scores.
    """
    from sklearn.model_selection import StratifiedKFold
    import copy

    cv_folds = config.get("training", {}).get("cv_folds", 5)
    min_class = min(dataset.class_counts.values())
    effective_folds = min(cv_folds, min_class)

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    sequences = dataset.sequences
    labels = dataset.labels

    fold_metrics: Dict[str, List[float]] = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        logger.info("CV fold %d/%d ...", fold + 1, effective_folds)

        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # Fresh copies
        fp = copy.deepcopy(feature_pipeline)
        mw = copy.deepcopy(model_wrapper)

        X_train = fp.fit_transform(train_seqs)
        X_val = fp.transform(val_seqs)

        try:
            mw.fit(X_train, y_train)
            y_pred = mw.predict(X_val)
            y_proba = mw.predict_proba(X_val)
            metrics = compute_metrics(y_val, y_pred, y_proba)
        except Exception as e:
            logger.error("CV fold %d failed: %s", fold + 1, e)
            continue

        for k, v in metrics.items():
            if isinstance(v, float):
                fold_metrics.setdefault(k, []).append(v)

    summary = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in fold_metrics.items()
    }
    logger.info(
        "CV (%s) — F1: %.3f ± %.3f | ROC-AUC: %.3f ± %.3f",
        model_wrapper.name,
        summary.get("f1_weighted", {}).get("mean", 0),
        summary.get("f1_weighted", {}).get("std", 0),
        summary.get("roc_auc", {}).get("mean", 0),
        summary.get("roc_auc", {}).get("std", 0),
    )
    return fold_metrics
