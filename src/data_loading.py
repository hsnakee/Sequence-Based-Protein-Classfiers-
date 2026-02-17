"""
data_loading.py - High-level dataset assembly and train/val/test splitting.

Integrates FASTA loading with feature extraction and provides sklearn-
compatible dataset splits with stratification and imbalance handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from fasta_utils import ProteinRecord, load_fasta, load_fasta_pair, compute_sequence_stats
from utils import get_logger, set_global_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

class ProteinDataset:
    """Container for labelled protein sequence data.

    Attributes:
        records: Flat list of all ProteinRecord objects.
        labels: Integer label array aligned with records.
        label_names: Mapping from integer label to class name string.
        sequences: List of sequence strings (convenience view).
    """

    def __init__(
        self,
        records: List[ProteinRecord],
        labels: List[int],
        label_names: Optional[Dict[int, str]] = None,
    ) -> None:
        if len(records) != len(labels):
            raise ValueError(
                f"records ({len(records)}) and labels ({len(labels)}) must have equal length."
            )
        self.records = records
        self.labels = np.array(labels, dtype=np.int64)
        self.label_names = label_names or {}

    # ---- convenience properties ----

    @property
    def sequences(self) -> List[str]:
        """List of sequence strings."""
        return [r.sequence for r in self.records]

    @property
    def ids(self) -> List[str]:
        """List of sequence IDs."""
        return [r.seq_id for r in self.records]

    @property
    def n_classes(self) -> int:
        """Number of unique classes."""
        return int(np.unique(self.labels).shape[0])

    @property
    def class_counts(self) -> Dict[int, int]:
        """Per-class sample counts."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    @property
    def class_weights(self) -> Dict[int, float]:
        """Balanced class weights (sklearn convention)."""
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(self.labels)
        weights = compute_class_weight("balanced", classes=classes, y=self.labels)
        return dict(zip(classes.tolist(), weights.tolist()))

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        counts = self.class_counts
        lines = [f"ProteinDataset(n={len(self)})"]
        for lbl, cnt in sorted(counts.items()):
            name = self.label_names.get(lbl, f"class_{lbl}")
            lines.append(f"  {name}: {cnt} sequences")
        return "\n".join(lines)

    def subset(self, indices: np.ndarray) -> "ProteinDataset":
        """Return a new ProteinDataset with the given indices.

        Args:
            indices: Integer array of indices to select.

        Returns:
            Subset ProteinDataset.
        """
        return ProteinDataset(
            records=[self.records[i] for i in indices],
            labels=self.labels[indices].tolist(),
            label_names=self.label_names,
        )

    def summary(self) -> Dict[str, Any]:
        """Compute and return dataset summary statistics.

        Returns:
            Dictionary with sequence stats per class and overall.
        """
        stats: Dict[str, Any] = {"overall": compute_sequence_stats(self.records)}
        for lbl, name in sorted(self.label_names.items()):
            class_records = [r for r, l in zip(self.records, self.labels) if l == lbl]
            if class_records:
                stats[name] = compute_sequence_stats(class_records)
        return stats


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(
    path_a: Union[str, Path],
    path_b: Union[str, Path],
    config: Dict[str, Any],
) -> ProteinDataset:
    """Load two FASTA files and construct a labelled ProteinDataset.

    Args:
        path_a: FASTA file for class 0.
        path_b: FASTA file for class 1.
        config: Full project config dict (reads ``data`` sub-key).

    Returns:
        A :class:`ProteinDataset` containing all records with labels.
    """
    data_cfg = config.get("data", {})
    project_cfg = config.get("project", {})

    label_a = data_cfg.get("class_a_label", 0)
    label_b = data_cfg.get("class_b_label", 1)
    name_a = data_cfg.get("class_a_name", Path(path_a).stem)
    name_b = data_cfg.get("class_b_name", Path(path_b).stem)

    common_kwargs: Dict[str, Any] = {
        "min_length": data_cfg.get("min_length", 1),
        "max_length": data_cfg.get("max_length", None),
        "remove_duplicates": data_cfg.get("remove_duplicates", True),
        "valid_amino_acids": data_cfg.get("valid_amino_acids", "ACDEFGHIKLMNPQRSTVWY"),
        "replace_unknown": data_cfg.get("replace_unknown", False),
        "show_progress": True,
    }

    records_a, records_b, _ = load_fasta_pair(
        path_a, path_b, label_a=label_a, label_b=label_b, **common_kwargs
    )

    all_records = records_a + records_b
    all_labels = [label_a] * len(records_a) + [label_b] * len(records_b)

    dataset = ProteinDataset(
        records=all_records,
        labels=all_labels,
        label_names={label_a: name_a, label_b: name_b},
    )

    logger.info("\n%s", dataset)
    return dataset


def load_unknown_dataset(
    path_c: Union[str, Path],
    config: Dict[str, Any],
) -> List[ProteinRecord]:
    """Load FASTA C (sequences to classify) without labels.

    Args:
        path_c: Path to unknown-class FASTA.
        config: Full project config dict.

    Returns:
        List of ProteinRecord objects.
    """
    data_cfg = config.get("data", {})

    records = load_fasta(
        path_c,
        label=None,
        min_length=data_cfg.get("min_length", 1),
        max_length=data_cfg.get("max_length", None),
        remove_duplicates=False,  # keep all for prediction
        valid_amino_acids=data_cfg.get("valid_amino_acids", "ACDEFGHIKLMNPQRSTVWY"),
        replace_unknown=data_cfg.get("replace_unknown", False),
        show_progress=True,
    )
    logger.info("Loaded %d sequences for prediction from %s", len(records), path_c)
    return records


# ---------------------------------------------------------------------------
# Splitting utilities
# ---------------------------------------------------------------------------

class DataSplitter:
    """Manages train/validation/test splits with stratification.

    Supports:
    - Stratified hold-out split
    - Stratified k-fold cross-validation
    - Small-dataset fallback (LOO when k-fold is infeasible)

    Args:
        test_size: Fraction of data for test set.
        val_size: Fraction of training data for validation.
        cv_folds: Number of cross-validation folds.
        seed: Random seed.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        cv_folds: int = 5,
        seed: int = 42,
    ) -> None:
        self.test_size = test_size
        self.val_size = val_size
        self.cv_folds = cv_folds
        self.seed = seed

    def split(
        self,
        dataset: ProteinDataset,
    ) -> Tuple["ProteinDataset", "ProteinDataset", "ProteinDataset"]:
        """Perform stratified train/val/test split.

        For very small datasets (< 3 × cv_folds), falls back to a
        simple 80/10/10 random split without stratification.

        Args:
            dataset: Full labelled dataset.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        n = len(dataset)
        labels = dataset.labels

        # Safety check for minimum class sizes
        min_class = min(dataset.class_counts.values())
        stratify = labels if min_class >= 3 else None
        if stratify is None:
            logger.warning(
                "Stratification disabled: smallest class has %d samples.", min_class
            )

        idx = np.arange(n)
        idx_trainval, idx_test = train_test_split(
            idx,
            test_size=self.test_size,
            stratify=stratify,
            random_state=self.seed,
        )

        stratify_tv = (
            labels[idx_trainval]
            if stratify is not None and min_class >= 4
            else None
        )
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=self.val_size / (1 - self.test_size),
            stratify=stratify_tv,
            random_state=self.seed,
        )

        logger.info(
            "Split: train=%d, val=%d, test=%d",
            len(idx_train),
            len(idx_val),
            len(idx_test),
        )

        return (
            dataset.subset(idx_train),
            dataset.subset(idx_val),
            dataset.subset(idx_test),
        )

    def cv_splits(
        self,
        dataset: ProteinDataset,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified k-fold cross-validation index pairs.

        Automatically reduces k if there are too few samples per class.

        Args:
            dataset: Full labelled dataset.

        Returns:
            List of (train_indices, val_indices) tuples.
        """
        min_class = min(dataset.class_counts.values())
        effective_folds = min(self.cv_folds, min_class)
        if effective_folds < self.cv_folds:
            logger.warning(
                "Reducing CV folds from %d to %d due to small class size.",
                self.cv_folds,
                effective_folds,
            )

        if effective_folds < 2:
            logger.warning("Cannot do CV with fewer than 2 folds; using single split.")
            idx = np.arange(len(dataset))
            split_point = max(1, int(0.8 * len(idx)))
            return [(idx[:split_point], idx[split_point:])]

        skf = StratifiedKFold(
            n_splits=effective_folds, shuffle=True, random_state=self.seed
        )
        return list(skf.split(np.zeros(len(dataset)), dataset.labels))
