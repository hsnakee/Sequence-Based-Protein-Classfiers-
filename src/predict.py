"""
predict.py - Prediction pipeline for classifying unknown sequences.

Loads trained artifacts, extracts features, and produces predictions
with class probabilities and confidence scores for FASTA C sequences.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from fasta_utils import ProteinRecord, load_fasta
from utils import get_logger, load_json

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prediction runner
# ---------------------------------------------------------------------------

class Predictor:
    """Loads trained artifacts and runs predictions on new sequences.

    Args:
        model_path: Path to a fitted classical model pickle file.
        feature_pipeline_path: Path to the fitted feature pipeline pickle.
        metadata_path: Path to the metadata JSON written by Trainer.
        threshold: Decision threshold for positive class.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        feature_pipeline_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        feature_pipeline_path = Path(feature_pipeline_path)

        logger.info("Loading model from %s", model_path)
        with open(model_path, "rb") as fh:
            self._model = pickle.load(fh)

        logger.info("Loading feature pipeline from %s", feature_pipeline_path)
        with open(feature_pipeline_path, "rb") as fh:
            self._feature_pipeline = pickle.load(fh)

        self.threshold = threshold
        self._class_labels: Dict[int, str] = {}

        if metadata_path is not None:
            metadata = load_json(metadata_path)
            self._class_labels = {int(k): v for k, v in metadata.get("class_labels", {}).items()}
            self.threshold = metadata.get("threshold", threshold)
            logger.info(
                "Loaded metadata: classes=%s, threshold=%.3f",
                self._class_labels, self.threshold,
            )

    def predict_sequences(self, sequences: List[str]) -> pd.DataFrame:
        """Classify a list of sequences.

        Args:
            sequences: List of amino acid strings.

        Returns:
            DataFrame with columns: sequence_index, predicted_class,
            class_name, probability_class0, probability_class1, confidence.
        """
        logger.info("Extracting features for %d sequences...", len(sequences))
        X = self._feature_pipeline.transform(sequences)

        logger.info("Running predictions...")
        proba = self._model.predict_proba(X)

        if proba.shape[1] == 2:
            y_pred = (proba[:, 1] >= self.threshold).astype(int)
        else:
            y_pred = proba.argmax(axis=1)

        rows = []
        for i, (pred, prob_vec) in enumerate(zip(y_pred, proba)):
            confidence = prob_vec.max()
            row: Dict[str, Any] = {
                "sequence_index": i,
                "predicted_class": int(pred),
                "class_name": self._class_labels.get(int(pred), f"class_{pred}"),
                "confidence": float(confidence),
            }
            for j, p in enumerate(prob_vec):
                name = self._class_labels.get(j, f"class_{j}")
                row[f"prob_{name}"] = float(p)
            rows.append(row)

        return pd.DataFrame(rows)

    def predict_fasta(
        self,
        fasta_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Load a FASTA file and classify all sequences.

        Args:
            fasta_path: Path to FASTA file.
            config: Optional project config dict (used for filtering).

        Returns:
            DataFrame with predictions plus seq_id and description columns.
        """
        data_cfg = config.get("data", {}) if config else {}
        records = load_fasta(
            fasta_path,
            min_length=data_cfg.get("min_length", 1),
            max_length=data_cfg.get("max_length", None),
            remove_duplicates=False,
            valid_amino_acids=data_cfg.get("valid_amino_acids", "ACDEFGHIKLMNPQRSTVWY"),
            replace_unknown=data_cfg.get("replace_unknown", False),
        )

        sequences = [r.sequence for r in records]
        results_df = self.predict_sequences(sequences)

        results_df.insert(0, "seq_id", [r.seq_id for r in records])
        results_df.insert(1, "description", [r.description for r in records])
        results_df.insert(2, "sequence_length", [len(r.sequence) for r in records])

        return results_df


# ---------------------------------------------------------------------------
# Standalone prediction function
# ---------------------------------------------------------------------------

def run_prediction(
    model_path: Union[str, Path],
    feature_pipeline_path: Union[str, Path],
    fasta_path: Union[str, Path],
    output_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """End-to-end prediction: load model → extract features → predict → save CSV.

    Args:
        model_path: Path to fitted classifier pickle.
        feature_pipeline_path: Path to fitted feature pipeline pickle.
        fasta_path: Path to FASTA file with unknown sequences.
        output_path: Path for output CSV file.
        metadata_path: Path to training metadata JSON (optional).
        config: Project config dict (optional).
        threshold: Decision threshold for positive class.

    Returns:
        DataFrame with predictions.
    """
    predictor = Predictor(
        model_path=model_path,
        feature_pipeline_path=feature_pipeline_path,
        metadata_path=metadata_path,
        threshold=threshold,
    )

    results = predictor.predict_fasta(fasta_path, config=config)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    pos_count = (results["predicted_class"] == 1).sum()
    neg_count = (results["predicted_class"] == 0).sum()
    logger.info(
        "Predictions saved to %s | Total: %d | Class0: %d | Class1: %d",
        output_path, len(results), neg_count, pos_count,
    )

    return results
