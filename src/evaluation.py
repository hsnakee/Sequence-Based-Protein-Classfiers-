"""
evaluation.py - Comprehensive model evaluation metrics and visualisations.

Implements:
- Full metric suite: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MCC
- Cross-validation evaluation
- Calibration curves
- Confusion matrices
- ROC and PR curves
- Feature importance plots
- Model comparison tables
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from utils import get_logger, ensure_dir

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    label_names: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (shape: n_samples × n_classes).
        threshold: Decision threshold (only affects binary classification).
        label_names: Mapping from int label to class name string.

    Returns:
        Dictionary of metric name → float value.
    """
    # Apply threshold for binary case
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_pred = (y_proba[:, 1] >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    # ROC-AUC and PR-AUC (binary)
    if y_proba is not None:
        try:
            n_classes = y_proba.shape[1] if y_proba.ndim == 2 else 2
            if n_classes == 2:
                pos_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = roc_auc_score(y_true, pos_proba)
                metrics["pr_auc"] = average_precision_score(y_true, pos_proba)
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
                metrics["pr_auc"] = average_precision_score(
                    y_true, y_proba, average="weighted"
                )
        except Exception as e:
            logger.warning("Could not compute AUC metrics: %s", e)

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find the probability threshold maximising a given metric.

    Args:
        y_true: Ground-truth binary labels.
        y_proba_pos: Predicted probabilities for the positive class.
        metric: One of "f1", "precision", "recall", "mcc".

    Returns:
        Tuple of (optimal_threshold, best_metric_value).
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh, best_score = 0.5, 0.0

    for t in thresholds:
        preds = (y_proba_pos >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, preds, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, preds, zero_division=0)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, preds)
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Evaluate and compare multiple classifier models.

    Args:
        output_dir: Directory to save plots and tables.
        label_names: Mapping from int label to class name.
        show_plots: Whether to display plots interactively.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "results",
        label_names: Optional[Dict[int, str]] = None,
        show_plots: bool = False,
        plot_format: str = "png",
        dpi: int = 150,
    ) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.label_names = label_names or {}
        self.show_plots = show_plots
        self.plot_format = plot_format
        self.dpi = dpi
        self._results: List[Dict[str, Any]] = []

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        feature_importances: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate a single model and record results.

        Args:
            model_name: Display name for the model.
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).
            feature_names: Feature name list (for importance plot).
            feature_importances: Feature importance array (for importance plot).

        Returns:
            Dictionary of computed metrics.
        """
        metrics = compute_metrics(y_true, y_pred, y_proba)
        metrics["model"] = model_name  # type: ignore[assignment]

        logger.info(
            "%-25s | Acc=%.3f | F1=%.3f | MCC=%.3f | ROC-AUC=%.3f",
            model_name,
            metrics.get("accuracy", 0),
            metrics.get("f1_weighted", 0),
            metrics.get("mcc", 0),
            metrics.get("roc_auc", 0),
        )

        self._results.append({"name": model_name, "metrics": metrics})

        # --- Plots ---
        try:
            self._plot_confusion_matrix(model_name, y_true, y_pred)
        except Exception as e:
            logger.warning("Confusion matrix plot failed: %s", e)

        if y_proba is not None and y_proba.ndim == 2:
            try:
                self._plot_roc(model_name, y_true, y_proba[:, 1])
            except Exception as e:
                logger.warning("ROC plot failed: %s", e)
            try:
                self._plot_pr(model_name, y_true, y_proba[:, 1])
            except Exception as e:
                logger.warning("PR plot failed: %s", e)
            try:
                self._plot_calibration(model_name, y_true, y_proba[:, 1])
            except Exception as e:
                logger.warning("Calibration plot failed: %s", e)

        if feature_importances is not None and feature_names is not None:
            try:
                self._plot_feature_importance(model_name, feature_names, feature_importances)
            except Exception as e:
                logger.warning("Feature importance plot failed: %s", e)

        return metrics

    def comparison_table(self) -> pd.DataFrame:
        """Return a sorted DataFrame comparing all evaluated models.

        Returns:
            DataFrame with one row per model and columns for all metrics.
        """
        if not self._results:
            return pd.DataFrame()

        rows = []
        for r in self._results:
            row = {"Model": r["name"]}
            row.update({k: v for k, v in r["metrics"].items() if k != "model"})
            rows.append(row)

        df = pd.DataFrame(rows)
        metric_cols = [c for c in df.columns if c not in ("Model",)]
        if "f1_weighted" in df.columns:
            df = df.sort_values("f1_weighted", ascending=False)

        return df.reset_index(drop=True)

    def save_comparison_table(self, filename: str = "model_comparison.csv") -> Path:
        """Save comparison table to CSV.

        Args:
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        df = self.comparison_table()
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        logger.info("Model comparison table saved to %s", path)
        return path

    # ---- Private plotting helpers ----

    def _get_axes(self, model_name: str, suffix: str) -> Tuple[Any, Any]:
        """Create a matplotlib figure and axes."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        return fig, ax

    def _save_or_show(self, fig: Any, model_name: str, suffix: str) -> None:
        import matplotlib.pyplot as plt
        fname = f"{model_name.replace(' ', '_')}_{suffix}.{self.plot_format}"
        path = self.output_dir / fname
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        logger.debug("Saved plot: %s", path)
        if self.show_plots:
            plt.show()
        plt.close(fig)

    def _plot_confusion_matrix(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred)
        labels = [self.label_names.get(i, str(i)) for i in sorted(set(y_true))]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix: {model_name}")
        plt.colorbar(im, ax=ax)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        self._save_or_show(fig, model_name, "confusion_matrix")

    def _plot_roc(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_proba_pos: np.ndarray,
    ) -> None:
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
        auc = roc_auc_score(y_true, y_proba_pos)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", lw=2, color="#2196F3")
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve: {model_name}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        self._save_or_show(fig, model_name, "roc_curve")

    def _plot_pr(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_proba_pos: np.ndarray,
    ) -> None:
        import matplotlib.pyplot as plt

        precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
        ap = average_precision_score(y_true, y_proba_pos)
        baseline = y_true.mean()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, label=f"AP = {ap:.3f}", lw=2, color="#4CAF50")
        ax.axhline(baseline, linestyle="--", color="gray", alpha=0.6, label=f"Baseline = {baseline:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve: {model_name}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        self._save_or_show(fig, model_name, "pr_curve")

    def _plot_calibration(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_proba_pos: np.ndarray,
    ) -> None:
        import matplotlib.pyplot as plt

        prob_true, prob_pred = calibration_curve(y_true, y_proba_pos, n_bins=10)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(prob_pred, prob_true, "s-", lw=2, color="#FF5722", label=model_name)
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration Curve: {model_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        self._save_or_show(fig, model_name, "calibration")

    def _plot_feature_importance(
        self,
        model_name: str,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 25,
    ) -> None:
        import matplotlib.pyplot as plt

        indices = np.argsort(importances)[::-1][:top_n]
        top_names = [feature_names[i] for i in indices]
        top_values = importances[indices]

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
        ax.barh(range(len(top_names)), top_values[::-1], color="#9C27B0")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top-{top_n} Feature Importances: {model_name}")
        ax.grid(axis="x", alpha=0.3)
        self._save_or_show(fig, model_name, "feature_importance")

    def plot_model_comparison(self, metric: str = "f1_weighted") -> None:
        """Plot a bar chart comparing all models on a given metric.

        Args:
            metric: Column name of the metric to compare.
        """
        import matplotlib.pyplot as plt

        df = self.comparison_table()
        if df.empty or metric not in df.columns:
            logger.warning("No data to plot for metric: %s", metric)
            return

        fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.2), 5))
        bars = ax.bar(df["Model"], df[metric], color="#2196F3", edgecolor="white")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.set_title(f"Model Comparison: {metric}")
        ax.set_xticklabels(df["Model"], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8,
            )

        fname = f"model_comparison_{metric}.{self.plot_format}"
        fig.tight_layout()
        fig.savefig(self.output_dir / fname, dpi=self.dpi)
        if self.show_plots:
            plt.show()
        plt.close(fig)
        logger.info("Model comparison plot saved to %s", self.output_dir / fname)


# ---------------------------------------------------------------------------
# Embedding visualisation
# ---------------------------------------------------------------------------

def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[Dict[int, str]] = None,
    output_dir: Union[str, Path] = "results",
    method: str = "pca",
    title: str = "Embedding Visualisation",
    show_plots: bool = False,
    plot_format: str = "png",
    dpi: int = 150,
) -> None:
    """Visualise high-dimensional embeddings using PCA or UMAP.

    Args:
        embeddings: 2D array of shape (n_samples, n_dim).
        labels: 1D integer label array.
        label_names: Mapping from int label to name string.
        output_dir: Directory to save the plot.
        method: "pca" or "umap".
        title: Plot title.
        show_plots: Display plot interactively.
        plot_format: Image format ("png", "pdf", etc.).
        dpi: Image resolution.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    output_dir = ensure_dir(output_dir)

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        explained = reducer.explained_variance_ratio_
        ax_labels = (
            f"PC1 ({explained[0]*100:.1f}%)",
            f"PC2 ({explained[1]*100:.1f}%)",
        )
    elif method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
            ax_labels = ("UMAP-1", "UMAP-2")
        except ImportError:
            logger.warning("umap-learn not installed; falling back to PCA.")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
            ax_labels = ("PC1", "PC2")
            method = "pca"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'umap'.")

    unique_labels = np.unique(labels)
    colours = cm.tab10(np.linspace(0, 1, len(unique_labels)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, colour in zip(unique_labels, colours):
        mask = labels == lbl
        name = label_names.get(int(lbl), str(lbl)) if label_names else str(lbl)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colour], label=name, alpha=0.7, s=20, edgecolors="none",
        )

    ax.set_xlabel(ax_labels[0])
    ax.set_ylabel(ax_labels[1])
    ax.set_title(title)
    ax.legend(loc="best", markerscale=2)
    ax.grid(alpha=0.3)

    fname = f"embedding_{method}.{plot_format}"
    fig.tight_layout()
    fig.savefig(output_dir / fname, dpi=dpi)
    if show_plots:
        plt.show()
    plt.close(fig)
    logger.info("Embedding plot saved to %s", output_dir / fname)
