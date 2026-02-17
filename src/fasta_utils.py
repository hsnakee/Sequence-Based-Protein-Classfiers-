"""
fasta_utils.py - FASTA file parsing and sequence preprocessing utilities.

Provides a robust multi-FASTA parser with support for:
- Multi-line sequences
- Invalid character filtering and uppercase normalisation
- Duplicate sequence / ID removal
- Optional length filtering
- Sequence identity clustering stub (CD-HIT-style grouping)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Standard amino acid alphabet (20 canonical AAs)
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
# Extended alphabet including ambiguous codes
EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYBUOZJX*-")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ProteinRecord:
    """Represents a single protein sequence record.

    Attributes:
        seq_id: Sequence identifier (everything after '>' up to first space).
        description: Full header line after '>'.
        sequence: Uppercase amino acid sequence (cleaned).
        original_length: Length before any truncation.
    """

    __slots__ = ("seq_id", "description", "sequence", "original_length")

    def __init__(
        self,
        seq_id: str,
        description: str,
        sequence: str,
    ) -> None:
        self.seq_id = seq_id
        self.description = description
        self.original_length = len(sequence)
        self.sequence = sequence

    def __repr__(self) -> str:
        return (
            f"ProteinRecord(id={self.seq_id!r}, "
            f"len={len(self.sequence)}, "
            f"seq={self.sequence[:20]!r}{'...' if len(self.sequence) > 20 else ''})"
        )

    def __len__(self) -> int:
        return len(self.sequence)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProteinRecord):
            return NotImplemented
        return self.sequence == other.sequence

    def __hash__(self) -> int:
        return hash(self.sequence)


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_fasta(
    path: Union[str, Path],
) -> Iterator[Tuple[str, str, str]]:
    """Low-level streaming FASTA parser. Yields (seq_id, description, raw_seq).

    Handles:
    - Multi-line sequences
    - Windows (\\r\\n) and Unix (\\n) line endings
    - Empty lines within sequences

    Args:
        path: Path to FASTA file.

    Yields:
        Tuples of (seq_id, full_description, concatenated_sequence).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no valid FASTA records are found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    current_id: Optional[str] = None
    current_desc: str = ""
    current_seq_parts: List[str] = []
    n_records = 0

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")
            if not line or line.startswith(";"):
                continue  # skip blank lines and FASTA comments

            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, current_desc, "".join(current_seq_parts)
                    n_records += 1
                header = line[1:].strip()
                # ID is everything up to first whitespace
                parts = header.split(None, 1)
                current_id = parts[0] if parts else "unknown"
                current_desc = header
                current_seq_parts = []
            else:
                if current_id is None:
                    logger.warning(
                        "Sequence data found before any header in %s; skipping.", path
                    )
                    continue
                current_seq_parts.append(line.strip())

    # Yield the last record
    if current_id is not None:
        yield current_id, current_desc, "".join(current_seq_parts)
        n_records += 1

    if n_records == 0:
        raise ValueError(
            f"No valid FASTA records found in {path}. "
            "Ensure the file starts with '>' header lines."
        )

    logger.debug("Parsed %d raw records from %s", n_records, path)


# ---------------------------------------------------------------------------
# Sequence cleaning
# ---------------------------------------------------------------------------

_NON_AA_PATTERN = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYBZUOJX]")


def clean_sequence(
    sequence: str,
    valid_chars: str = "ACDEFGHIKLMNPQRSTVWY",
    replace_unknown: bool = False,
    unknown_char: str = "X",
) -> str:
    """Normalise and clean a protein sequence.

    Steps:
    1. Uppercase
    2. Remove whitespace and digits
    3. Optionally replace non-standard residues with ``unknown_char``
    4. If not replacing, remove non-standard characters

    Args:
        sequence: Raw amino acid string.
        valid_chars: Set of valid amino acid single-letter codes.
        replace_unknown: If True, replace invalid chars; else remove them.
        unknown_char: Replacement character when ``replace_unknown=True``.

    Returns:
        Cleaned sequence string.
    """
    seq = sequence.upper().replace(" ", "").replace("\t", "")
    seq = re.sub(r"[0-9\-\*]", "", seq)  # remove digits, gaps, stops

    valid_set = set(valid_chars.upper())
    if replace_unknown:
        cleaned = "".join(c if c in valid_set else unknown_char for c in seq)
    else:
        cleaned = "".join(c for c in seq if c in valid_set)

    return cleaned


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

def load_fasta(
    path: Union[str, Path],
    label: Optional[int] = None,
    min_length: int = 1,
    max_length: Optional[int] = None,
    remove_duplicates: bool = True,
    valid_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY",
    replace_unknown: bool = False,
    unknown_char: str = "X",
    show_progress: bool = True,
) -> List[ProteinRecord]:
    """Parse a FASTA file and return cleaned ``ProteinRecord`` objects.

    Args:
        path: Path to the FASTA file.
        label: Optional integer label to attach (unused in record itself).
        min_length: Discard sequences shorter than this after cleaning.
        max_length: Truncate sequences longer than this (None = no limit).
        remove_duplicates: Remove records with identical sequences.
        valid_amino_acids: Allowed amino acid characters.
        replace_unknown: Replace invalid chars instead of removing them.
        unknown_char: Replacement character for invalid residues.
        show_progress: Show tqdm progress bar while loading.

    Returns:
        List of ``ProteinRecord`` objects, filtered and cleaned.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If no valid sequences remain after filtering.
    """
    path = Path(path)
    logger.info("Loading FASTA: %s", path)

    records: List[ProteinRecord] = []
    seen_sequences: set[str] = set()
    stats: Dict[str, int] = defaultdict(int)

    raw_iter = parse_fasta(path)
    if show_progress:
        raw_iter = tqdm(raw_iter, desc=f"Parsing {path.name}", unit="seq", leave=False)  # type: ignore[assignment]

    for seq_id, description, raw_seq in raw_iter:
        stats["total"] += 1

        # Clean sequence
        cleaned = clean_sequence(
            raw_seq,
            valid_chars=valid_amino_acids,
            replace_unknown=replace_unknown,
            unknown_char=unknown_char,
        )

        if len(cleaned) == 0:
            stats["empty_after_clean"] += 1
            logger.debug("Skipping %s: empty after cleaning.", seq_id)
            continue

        # Length filter
        if len(cleaned) < min_length:
            stats["too_short"] += 1
            continue

        # Truncate if needed
        if max_length is not None and len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
            stats["truncated"] += 1

        # Duplicate removal
        if remove_duplicates:
            if cleaned in seen_sequences:
                stats["duplicates"] += 1
                logger.debug("Skipping duplicate: %s", seq_id)
                continue
            seen_sequences.add(cleaned)

        records.append(ProteinRecord(seq_id=seq_id, description=description, sequence=cleaned))
        stats["kept"] += 1

    logger.info(
        "FASTA %s — Total: %d | Kept: %d | Duplicates: %d | "
        "Too short: %d | Empty: %d | Truncated: %d",
        path.name,
        stats["total"],
        stats["kept"],
        stats["duplicates"],
        stats["too_short"],
        stats["empty_after_clean"],
        stats["truncated"],
    )

    if not records:
        raise ValueError(
            f"No valid sequences remain from {path} after filtering. "
            f"Check min_length ({min_length}) and valid_amino_acids settings."
        )

    return records


# ---------------------------------------------------------------------------
# Multiple FASTA loading
# ---------------------------------------------------------------------------

def load_fasta_pair(
    path_a: Union[str, Path],
    path_b: Union[str, Path],
    label_a: int = 0,
    label_b: int = 1,
    **kwargs: Any,
) -> Tuple[List[ProteinRecord], List[ProteinRecord], List[int]]:
    """Load two FASTA files and return records + labels.

    Args:
        path_a: Path to class A FASTA.
        path_b: Path to class B FASTA.
        label_a: Integer label for class A.
        label_b: Integer label for class B.
        **kwargs: Additional arguments forwarded to :func:`load_fasta`.

    Returns:
        Tuple of (records_a, records_b, labels) where labels corresponds
        to the concatenation of both record lists.
    """
    records_a = load_fasta(path_a, label=label_a, **kwargs)
    records_b = load_fasta(path_b, label=label_b, **kwargs)

    labels = [label_a] * len(records_a) + [label_b] * len(records_b)

    logger.info(
        "Class A (%s): %d sequences | Class B (%s): %d sequences",
        Path(path_a).name,
        len(records_a),
        Path(path_b).name,
        len(records_b),
    )

    imbalance_ratio = len(records_a) / max(len(records_b), 1)
    if imbalance_ratio > 3 or imbalance_ratio < 0.33:
        logger.warning(
            "Significant class imbalance detected: %.1f:1 ratio. "
            "Consider using class_weight='balanced' or SMOTE.",
            max(imbalance_ratio, 1 / imbalance_ratio),
        )

    return records_a, records_b, labels


def write_fasta(
    records: List[ProteinRecord],
    path: Union[str, Path],
    line_width: int = 60,
) -> None:
    """Write ProteinRecord objects to a FASTA file.

    Args:
        records: List of ProteinRecord objects.
        path: Output file path.
        line_width: Characters per sequence line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(f">{rec.description}\n")
            seq = rec.sequence
            for i in range(0, len(seq), line_width):
                fh.write(seq[i : i + line_width] + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


# ---------------------------------------------------------------------------
# Sequence statistics
# ---------------------------------------------------------------------------

def compute_sequence_stats(records: List[ProteinRecord]) -> Dict[str, float]:
    """Compute summary statistics for a collection of sequences.

    Args:
        records: List of ProteinRecord objects.

    Returns:
        Dictionary with keys: n_sequences, min_length, max_length,
        mean_length, median_length, std_length.
    """
    import statistics

    lengths = [len(r) for r in records]
    return {
        "n_sequences": len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
        "std_length": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Type annotation shim
# ---------------------------------------------------------------------------
from typing import Any  # noqa: E402  (needed for kwargs hint)
