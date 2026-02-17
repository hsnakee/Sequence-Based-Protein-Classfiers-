"""
feature_extractors.py - Modular, pluggable feature extraction pipelines.

Implements:
- One-hot encoding with configurable padding/truncation
- k-mer frequency vectors (configurable k, multiple k simultaneously)
- Amino acid composition (1-gram, dipeptide)
- Physicochemical property features
- Sequence-level global statistics
- Feature pipeline registry and composition utilities
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from fasta_utils import ProteinRecord, STANDARD_AA
from utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMINO_ACIDS = sorted(STANDARD_AA)           # 20 canonical AAs in sorted order
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY: Dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Net charge at pH 7.4 (simplified)
CHARGE_AT_PH7: Dict[str, float] = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0.1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

# Molecular weight (Da) of each residue
MOLECULAR_WEIGHT: Dict[str, float] = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

# Aromaticity flag
AROMATICITY: Dict[str, float] = {
    "A": 0, "R": 0, "N": 0, "D": 0, "C": 0,
    "Q": 0, "E": 0, "G": 0, "H": 0, "I": 0,
    "L": 0, "K": 0, "M": 0, "F": 1, "P": 0,
    "S": 0, "T": 0, "W": 1, "Y": 1, "V": 0,
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseFeatureExtractor(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for all feature extractors.

    All extractors must implement :meth:`fit` and :meth:`transform` and
    expose a :attr:`feature_names` property.
    """

    @abstractmethod
    def fit(self, sequences: List[str], y: Optional[np.ndarray] = None) -> "BaseFeatureExtractor":
        """Fit the extractor (compute stats, vocabulary, etc.).

        Args:
            sequences: List of amino acid sequence strings.
            y: Ignored; present for sklearn compatibility.

        Returns:
            Self.
        """
        ...

    @abstractmethod
    def transform(self, sequences: List[str]) -> np.ndarray:
        """Transform sequences into feature matrix.

        Args:
            sequences: List of amino acid sequence strings.

        Returns:
            2D numpy array of shape (n_samples, n_features).
        """
        ...

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Human-readable feature names."""
        ...

    @property
    def n_features(self) -> int:
        """Number of output features."""
        return len(self.feature_names)

    def fit_transform(self, sequences: List[str], y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(sequences, y).transform(sequences)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_features={self.n_features})"


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

class OneHotExtractor(BaseFeatureExtractor):
    """Encode sequences as flattened one-hot matrices with padding/truncation.

    Args:
        max_length: Fixed sequence length for padding/truncation.
            If None, uses the maximum training sequence length.
        flatten: If True, return a 1D vector per sequence.
            If False, return shape (max_length, 20).
    """

    def __init__(self, max_length: Optional[int] = None, flatten: bool = True) -> None:
        self.max_length = max_length
        self.flatten = flatten
        self._max_length_fitted: Optional[int] = None

    def fit(self, sequences: List[str], y: Optional[np.ndarray] = None) -> "OneHotExtractor":
        if self.max_length is not None:
            self._max_length_fitted = self.max_length
        else:
            self._max_length_fitted = max(len(s) for s in sequences)
            logger.debug("OneHotExtractor: inferred max_length=%d", self._max_length_fitted)
        return self

    def transform(self, sequences: List[str]) -> np.ndarray:
        if self._max_length_fitted is None:
            raise RuntimeError("Call fit() before transform().")
        L = self._max_length_fitted
        n = len(sequences)
        matrix = np.zeros((n, L, 20), dtype=np.float32)

        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[:L]):
                idx = AA_TO_IDX.get(aa, None)
                if idx is not None:
                    matrix[i, j, idx] = 1.0

        if self.flatten:
            return matrix.reshape(n, -1)
        return matrix

    @property
    def feature_names(self) -> List[str]:
        if self._max_length_fitted is None:
            return []
        L = self._max_length_fitted
        if self.flatten:
            return [f"oh_{pos}_{aa}" for pos in range(L) for aa in AMINO_ACIDS]
        return [f"pos{p}_{aa}" for p in range(L) for aa in AMINO_ACIDS]


# ---------------------------------------------------------------------------
# k-mer frequency vectors
# ---------------------------------------------------------------------------

class KmerExtractor(BaseFeatureExtractor):
    """Compute k-mer frequency feature vectors.

    Supports multiple simultaneous k values; all are concatenated.

    Args:
        k: Single integer or list of integers (e.g. [2, 3, 4]).
        normalize: If True, convert counts to frequencies (sum to 1 per seq).
    """

    def __init__(self, k: Union[int, List[int]] = 3, normalize: bool = True) -> None:
        self.k = k if isinstance(k, list) else [k]
        self.normalize = normalize
        self._vocabularies: Dict[int, List[str]] = {}

    def _build_vocab(self, k_val: int) -> List[str]:
        return ["".join(p) for p in itertools.product(AMINO_ACIDS, repeat=k_val)]

    def fit(self, sequences: List[str], y: Optional[np.ndarray] = None) -> "KmerExtractor":
        for k_val in self.k:
            self._vocabularies[k_val] = self._build_vocab(k_val)
        logger.debug(
            "KmerExtractor: k=%s, total features=%d", self.k, self.n_features
        )
        return self

    def _count_kmers(self, sequence: str, k_val: int) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for i in range(len(sequence) - k_val + 1):
            kmer = sequence[i : i + k_val]
            if len(kmer) == k_val:
                counts[kmer] = counts.get(kmer, 0) + 1
        return counts

    def transform(self, sequences: List[str]) -> np.ndarray:
        if not self._vocabularies:
            raise RuntimeError("Call fit() before transform().")

        rows = []
        for seq in sequences:
            row_parts = []
            for k_val in self.k:
                vocab = self._vocabularies[k_val]
                counts = self._count_kmers(seq, k_val)
                vec = np.array([counts.get(km, 0) for km in vocab], dtype=np.float32)
                if self.normalize:
                    total = vec.sum()
                    if total > 0:
                        vec /= total
                row_parts.append(vec)
            rows.append(np.concatenate(row_parts))

        return np.vstack(rows)

    @property
    def feature_names(self) -> List[str]:
        names = []
        for k_val in self.k:
            vocab = self._vocabularies.get(k_val, self._build_vocab(k_val))
            names.extend([f"kmer{k_val}_{km}" for km in vocab])
        return names


# ---------------------------------------------------------------------------
# Amino acid composition
# ---------------------------------------------------------------------------

class CompositionExtractor(BaseFeatureExtractor):
    """Compute amino acid composition (and optionally dipeptide composition).

    Args:
        include_dipeptide: If True, add 20×20=400 dipeptide frequency features.
    """

    def __init__(self, include_dipeptide: bool = False) -> None:
        self.include_dipeptide = include_dipeptide
        self._dipeptide_vocab: Optional[List[str]] = None

    def fit(self, sequences: List[str], y: Optional[np.ndarray] = None) -> "CompositionExtractor":
        if self.include_dipeptide:
            self._dipeptide_vocab = [
                f"{a}{b}" for a in AMINO_ACIDS for b in AMINO_ACIDS
            ]
        return self

    def transform(self, sequences: List[str]) -> np.ndarray:
        rows = []
        for seq in sequences:
            # Monomer composition (normalised)
            mono = np.zeros(20, dtype=np.float32)
            for aa in seq:
                idx = AA_TO_IDX.get(aa)
                if idx is not None:
                    mono[idx] += 1
            total = max(mono.sum(), 1)
            mono /= total

            if self.include_dipeptide and self._dipeptide_vocab is not None:
                di_counts: Dict[str, int] = {}
                for i in range(len(seq) - 1):
                    dp = seq[i : i + 2]
                    di_counts[dp] = di_counts.get(dp, 0) + 1
                di_total = max(sum(di_counts.values()), 1)
                di_vec = np.array(
                    [di_counts.get(dp, 0) / di_total for dp in self._dipeptide_vocab],
                    dtype=np.float32,
                )
                rows.append(np.concatenate([mono, di_vec]))
            else:
                rows.append(mono)

        return np.vstack(rows)

    @property
    def feature_names(self) -> List[str]:
        names = [f"comp_{aa}" for aa in AMINO_ACIDS]
        if self.include_dipeptide and self._dipeptide_vocab:
            names += [f"dipep_{dp}" for dp in self._dipeptide_vocab]
        return names


# ---------------------------------------------------------------------------
# Physicochemical features
# ---------------------------------------------------------------------------

class PhysicochemicalExtractor(BaseFeatureExtractor):
    """Compute physicochemical property statistics per sequence.

    For each property (hydrophobicity, charge, MW, aromaticity), computes:
    mean, std, min, max, and the fraction of residues above mean.

    Args:
        include_hydrophobicity: Include Kyte-Doolittle hydrophobicity.
        include_charge: Include net charge at pH 7.4.
        include_molecular_weight: Include residue MW statistics.
        include_aromaticity: Include fraction of aromatic residues.
    """

    def __init__(
        self,
        include_hydrophobicity: bool = True,
        include_charge: bool = True,
        include_molecular_weight: bool = True,
        include_aromaticity: bool = True,
    ) -> None:
        self.include_hydrophobicity = include_hydrophobicity
        self.include_charge = include_charge
        self.include_molecular_weight = include_molecular_weight
        self.include_aromaticity = include_aromaticity

    def fit(self, sequences: List[str], y: Optional[np.ndarray] = None) -> "PhysicochemicalExtractor":
        return self  # stateless

    def _property_stats(self, seq: str, prop_dict: Dict[str, float]) -> np.ndarray:
        vals = np.array([prop_dict.get(aa, 0.0) for aa in seq], dtype=np.float32)
        if len(vals) == 0:
            return np.zeros(5, dtype=np.float32)
        mean_val = vals.mean()
        return np.array([
            mean_val,
            vals.std() if len(vals) > 1 else 0.0,
            vals.min(),
            vals.max(),
            (vals > mean_val).mean(),  # fraction above mean
        ], dtype=np.float32)

    def transform(self, sequences: List[str]) -> np.ndarray:
        rows = []
        for seq in sequences:
            parts = []
            if self.include_hydrophobicity:
                parts.append(self._property_stats(seq, HYDROPHOBICITY))
            if self.include_charge:
                parts.append(self._property_stats(seq, CHARGE_AT_PH7))
            if self.include_molecular_weight:
                parts.append(self._property_stats(seq, MOLECULAR_WEIGHT))
            if self.include_aromaticity:
                # single value: fraction of aromatic residues
                arom = np.mean([AROMATICITY.get(aa, 0.0) for aa in seq]) if seq else 0.0
                parts.append(np.array([arom], dtype=np.float32))
            rows.append(np.concatenate(parts))
        return np.vstack(rows)

    @property
    def feature_names(self) -> List[str]:
        stat_names = ["mean", "std", "min", "max", "frac_above_mean"]
        names = []
        if self.include_hydrophobicity:
            names += [f"hydro_{s}" for s in stat_names]
        if self.include_charge:
            names += [f"charge_{s}" for s in stat_names]
        if self.include_molecular_weight:
            names += [f"mw_{s}" for s in stat_names]
        if self.include_aromaticity:
            names += ["aromaticity_frac"]
        return names


# ---------------------------------------------------------------------------
# Global sequence statistics
# ---------------------------------------------------------------------------

class GlobalStatsExtractor(BaseFeatureExtractor):
    """Compute sequence-level global statistics.

    Features: sequence length, log(length), and basic composition ratios.
    """

    def fit(self, sequences: List[str], y: Optional[np.ndarray] = None) -> "GlobalStatsExtractor":
        self._max_len = max(len(s) for s in sequences)
        return self

    def transform(self, sequences: List[str]) -> np.ndarray:
        rows = []
        for seq in sequences:
            length = len(seq)
            log_length = np.log1p(length)
            norm_length = length / max(self._max_len, 1)

            # Basic group fractions
            polar = sum(1 for aa in seq if aa in "NQST")
            nonpolar = sum(1 for aa in seq if aa in "AVILMFYW")
            charged_pos = sum(1 for aa in seq if aa in "RK")
            charged_neg = sum(1 for aa in seq if aa in "DE")
            n = max(length, 1)

            rows.append(np.array([
                length,
                log_length,
                norm_length,
                polar / n,
                nonpolar / n,
                charged_pos / n,
                charged_neg / n,
                (charged_pos - charged_neg) / n,  # net charge ratio
            ], dtype=np.float32))

        return np.vstack(rows)

    @property
    def feature_names(self) -> List[str]:
        return [
            "seq_length",
            "log_length",
            "norm_length",
            "frac_polar",
            "frac_nonpolar",
            "frac_charged_pos",
            "frac_charged_neg",
            "net_charge_ratio",
        ]


# ---------------------------------------------------------------------------
# Feature pipeline composer
# ---------------------------------------------------------------------------

class FeaturePipeline(BaseEstimator, TransformerMixin):
    """Compose multiple feature extractors and concatenate their outputs.

    Args:
        extractors: List of (name, extractor) tuples.
        scaler: sklearn scaler to apply after concatenation.
            Pass None to skip scaling.
    """

    def __init__(
        self,
        extractors: List[Tuple[str, BaseFeatureExtractor]],
        scaler: Optional[Any] = None,
    ) -> None:
        self.extractors = extractors
        self.scaler = scaler

    def fit(self, sequences: List[str], y: Optional[np.ndarray] = None) -> "FeaturePipeline":
        for name, ext in self.extractors:
            logger.debug("Fitting extractor: %s", name)
            ext.fit(sequences, y)
        if self.scaler is not None:
            raw = self._concatenate(sequences)
            self.scaler.fit(raw)
        return self

    def _concatenate(self, sequences: List[str]) -> np.ndarray:
        parts = []
        for name, ext in self.extractors:
            feat = ext.transform(sequences)
            parts.append(feat)
        return np.hstack(parts) if parts else np.zeros((len(sequences), 0))

    def transform(self, sequences: List[str]) -> np.ndarray:
        X = self._concatenate(sequences)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def fit_transform(self, sequences: List[str], y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(sequences, y).transform(sequences)

    @property
    def feature_names(self) -> List[str]:
        names = []
        for _, ext in self.extractors:
            names.extend(ext.feature_names)
        return names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def __repr__(self) -> str:
        parts = [name for name, _ in self.extractors]
        return f"FeaturePipeline([{', '.join(parts)}], n_features={self.n_features})"


# ---------------------------------------------------------------------------
# Factory / Registry
# ---------------------------------------------------------------------------

def build_feature_pipeline(config: Dict[str, Any]) -> FeaturePipeline:
    """Build a FeaturePipeline from the project config dict.

    Reads ``config['features']`` to determine which extractors to include.
    Excludes one-hot by default (too high-dimensional for classical ML);
    use ``include_onehot=True`` kwarg override if needed.

    Args:
        config: Full project config dict.

    Returns:
        An un-fitted :class:`FeaturePipeline`.
    """
    feat_cfg = config.get("features", {})
    training_cfg = config.get("training", {})
    scaler_name = training_cfg.get("feature_scaling", "standard")

    extractors: List[Tuple[str, BaseFeatureExtractor]] = []

    # k-mer
    kmer_cfg = feat_cfg.get("kmer", {})
    k_vals = kmer_cfg.get("k", [3])
    extractors.append((
        "kmer",
        KmerExtractor(k=k_vals, normalize=kmer_cfg.get("normalize", True)),
    ))

    # Composition
    comp_cfg = feat_cfg.get("composition", {})
    extractors.append((
        "composition",
        CompositionExtractor(include_dipeptide=comp_cfg.get("include_dipeptide", False)),
    ))

    # Physicochemical
    phys_cfg = feat_cfg.get("physicochemical", {})
    extractors.append((
        "physicochemical",
        PhysicochemicalExtractor(
            include_hydrophobicity=phys_cfg.get("include_hydrophobicity", True),
            include_charge=phys_cfg.get("include_charge", True),
            include_molecular_weight=phys_cfg.get("include_molecular_weight", True),
            include_aromaticity=phys_cfg.get("include_aromaticity", True),
        ),
    ))

    # Global stats
    extractors.append(("global_stats", GlobalStatsExtractor()))

    # Scaler
    scaler = None
    if scaler_name == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif scaler_name == "robust":
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()

    pipeline = FeaturePipeline(extractors=extractors, scaler=scaler)
    logger.info("Built feature pipeline with extractors: %s", [n for n, _ in extractors])
    return pipeline


def build_onehot_pipeline(config: Dict[str, Any]) -> FeaturePipeline:
    """Build a one-hot only feature pipeline (for neural models).

    Args:
        config: Full project config dict.

    Returns:
        Un-fitted :class:`FeaturePipeline` with only :class:`OneHotExtractor`.
    """
    oh_cfg = config.get("features", {}).get("one_hot", {})
    return FeaturePipeline(
        extractors=[("one_hot", OneHotExtractor(
            max_length=oh_cfg.get("max_length", None),
            flatten=oh_cfg.get("flatten", True),
        ))],
        scaler=None,
    )


# ---------------------------------------------------------------------------
# Type hint shim
# ---------------------------------------------------------------------------
from typing import Union  # noqa: E402
