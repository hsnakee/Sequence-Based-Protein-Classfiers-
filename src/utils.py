"""
utils.py - Shared utility functions for protein sequence classification.

Provides logging setup, seeding, config loading, device detection,
timing utilities, and other helpers used across the project.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> logging.Logger:
    """Configure root logger with console (and optional file) handlers.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to a file.
        fmt: Log message format string.

    Returns:
        Configured root logger.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=numeric_level, format=fmt, handlers=handlers, force=True)
    logger = logging.getLogger(__name__)
    logger.info("Logging initialised at level %s", level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and (if available) PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    get_logger(__name__).debug("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    get_logger(__name__).info("Loaded config from %s", config_path)
    return config


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely retrieve a nested value from a dict using dot-style key path.

    Args:
        config: Configuration dictionary.
        *keys: Sequence of string keys forming the path.
        default: Value to return if any key is missing.

    Returns:
        Value at the nested path, or ``default`` if not found.

    Example:
        >>> get_nested(cfg, "training", "cv_folds", default=5)
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> "torch.device":  # type: ignore[name-defined]
    """Determine the best available torch device.

    Args:
        preference: One of "auto", "cpu", "cuda", or "mps".

    Returns:
        A ``torch.device`` object.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for neural models.") from exc

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    # Auto-detect
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    get_logger(__name__).info("Using device: %s", device)
    return device


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for timing code blocks.

    Example:
        >>> with Timer("model training") as t:
        ...     model.fit(X, y)
        >>> print(t.elapsed)
    """

    def __init__(self, name: str = "", logger: Optional[logging.Logger] = None) -> None:
        self.name = name
        self.elapsed: float = 0.0
        self._logger = logger or get_logger(__name__)

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        self._logger.info(
            "⏱  %s completed in %.2fs", self.name or "Block", self.elapsed
        )


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def md5_of_file(path: Union[str, Path]) -> str:
    """Compute MD5 hash of a file for cache invalidation.

    Args:
        path: File path.

    Returns:
        Hex-string MD5 digest.
    """
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of_strings(strings: list[str]) -> str:
    """Compute MD5 hash of an ordered list of strings.

    Args:
        strings: List of strings to hash.

    Returns:
        Hex-string MD5 digest.
    """
    h = hashlib.md5()
    for s in strings:
        h.update(s.encode("utf-8"))
    return h.hexdigest()


def save_json(data: Any, path: Union[str, Path]) -> None:
    """Save arbitrary data as JSON.

    Args:
        data: JSON-serialisable object.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON from a file.

    Args:
        path: File path.

    Returns:
        Deserialised Python object.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_banner(title: str, width: int = 70) -> None:
    """Print a formatted banner to stdout.

    Args:
        title: Text to display in the banner.
        width: Total banner width in characters.
    """
    border = "=" * width
    padded = title.center(width - 2)
    print(f"\n{border}\n {padded}\n{border}\n")


def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Format a metrics dictionary as an ASCII table string.

    Args:
        metrics: Dictionary mapping metric names to float values.
        title: Table title.

    Returns:
        Formatted string table.
    """
    lines = [f"\n{'─' * 40}", f"  {title}", f"{'─' * 40}"]
    for name, value in sorted(metrics.items()):
        lines.append(f"  {name:<25} {value:.4f}")
    lines.append(f"{'─' * 40}\n")
    return "\n".join(lines)
