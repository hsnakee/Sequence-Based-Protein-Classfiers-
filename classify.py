#!/usr/bin/env python3
"""
classify.py - CLI entry point for classifying unknown protein sequences.

Usage:
    python classify.py \\
        --model results/classical_RandomForest.pkl \\
        --pipeline results/feature_pipeline.pkl \\
        --input data/unknown.fasta \\
        --output predictions.csv \\
        [--config configs/default.yaml] \\
        [--threshold 0.5]

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import run_prediction
from utils import load_config, setup_logging, print_banner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Classify unknown protein sequences using a trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, type=Path,
        help="Path to trained classifier pickle file.",
    )
    parser.add_argument(
        "--pipeline", required=True, type=Path,
        help="Path to fitted feature pipeline pickle file.",
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="FASTA file with unknown sequences to classify.",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output CSV file path for predictions.",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml", type=Path,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--metadata", default=None, type=Path,
        help="Path to training metadata JSON (for class names and threshold).",
    )
    parser.add_argument(
        "--threshold", default=0.5, type=float,
        help="Decision threshold for the positive class (default: 0.5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main classification entry point."""
    args = parse_args(argv)

    # Validate inputs
    for path, name in [
        (args.model, "model"),
        (args.pipeline, "pipeline"),
        (args.input, "input"),
    ]:
        if not path.exists():
            print(f"ERROR: {name} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    setup_logging(level="INFO")
    print_banner("Protein Sequence Classifier — Classify")

    # Load config (optional, for filtering params)
    config = None
    if args.config.exists():
        config = load_config(args.config)

    # Detect metadata path automatically if not provided
    metadata_path = args.metadata
    if metadata_path is None:
        auto_meta = args.model.parent / "metadata.json"
        if auto_meta.exists():
            metadata_path = auto_meta

    # Run prediction
    results = run_prediction(
        model_path=args.model,
        feature_pipeline_path=args.pipeline,
        fasta_path=args.input,
        output_path=args.output,
        metadata_path=metadata_path,
        config=config,
        threshold=args.threshold,
    )

    # Summary
    print(f"\n✅ Classification complete.")
    print(f"   Input sequences: {len(results)}")
    print(f"   Output file: {args.output}")

    if "class_name" in results.columns:
        counts = results["class_name"].value_counts()
        for cls, cnt in counts.items():
            pct = 100 * cnt / len(results)
            print(f"   {cls}: {cnt} ({pct:.1f}%)")

    if "confidence" in results.columns:
        print(f"   Mean confidence: {results['confidence'].mean():.3f}")


if __name__ == "__main__":
    main()
