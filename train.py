#!/usr/bin/env python3
"""
train.py - CLI entry point for training protein sequence classifiers.

Usage:
    python train.py \\
        --classA data/snare.fasta \\
        --classB data/non_snare.fasta \\
        --config configs/default.yaml \\
        --output results/ \\
        [--no-classical] [--no-neural] [--seed 42]

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add src/ to path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loading import build_dataset
from training import Trainer
from utils import load_config, setup_logging, set_global_seed, print_banner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train protein sequence classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--classA", required=True, type=Path,
        help="FASTA file for class 0 (e.g., positive/SNARE sequences).",
    )
    parser.add_argument(
        "--classB", required=True, type=Path,
        help="FASTA file for class 1 (e.g., negative/non-SNARE sequences).",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml", type=Path,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output", default=None, type=Path,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--seed", default=None, type=int,
        help="Override random seed from config.",
    )
    parser.add_argument(
        "--no-classical", action="store_true",
        help="Skip classical ML model training.",
    )
    parser.add_argument(
        "--no-neural", action="store_true",
        help="Skip neural network training.",
    )
    parser.add_argument(
        "--name-a", default=None,
        help="Display name for class A (overrides config).",
    )
    parser.add_argument(
        "--name-b", default=None,
        help="Display name for class B (overrides config).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main training entry point."""
    args = parse_args(argv)

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.output:
        config.setdefault("project", {})["output_dir"] = str(args.output)
    if args.seed is not None:
        config.setdefault("project", {})["seed"] = args.seed
    if args.name_a:
        config.setdefault("data", {})["class_a_name"] = args.name_a
    if args.name_b:
        config.setdefault("data", {})["class_b_name"] = args.name_b

    # Setup logging
    output_dir = Path(config.get("project", {}).get("output_dir", "results"))
    setup_logging(
        level=config.get("project", {}).get("log_level", "INFO"),
        log_file=output_dir / "train.log",
    )

    seed = config.get("project", {}).get("seed", 42)
    set_global_seed(seed)

    print_banner("Protein Sequence Classifier — Train")

    # Validate input files
    for path, name in [(args.classA, "classA"), (args.classB, "classB")]:
        if not path.exists():
            print(f"ERROR: {name} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Build dataset
    dataset = build_dataset(args.classA, args.classB, config)

    # Train
    trainer = Trainer(config)
    artifacts = trainer.run(
        dataset,
        train_classical=not args.no_classical,
        train_neural=not args.no_neural,
    )

    print(f"\n✅ Training complete. Results saved to: {output_dir}")
    print(f"   Best model: {artifacts.best_model_name}")


if __name__ == "__main__":
    main()
