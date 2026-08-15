"""Utility functions for analysis module."""

import io
import logging
import sys
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt

from .legacy_records import LegacyRecord

logger = logging.getLogger(__name__)


def compute_counting_cutoff(
    turns: list[LegacyRecord], max_turns: int, penalty: int
) -> int | None:
    """Compute turn number where score becomes guaranteed <= 0.

    After turn n with k cumulative failed guesses, best possible score is:
        max_turns - (n+1) - penalty*k
    When penalty*k >= max_turns - n, score is guaranteed <= 0.

    Args:
        turns: List of turn dicts from results.json (with turn_number, guess_attempt)
        max_turns: Maximum turns allowed in the game
        penalty: Points lost per failed guess

    Returns:
        Turn number where score became guaranteed <= 0, or None if never hit.
    """
    cumulative_failed = 0
    for turn in turns:
        turn_num = turn["turn_number"]  # 1-indexed

        guess_attempt = turn.get("guess_attempt")
        if (
            guess_attempt
            and not guess_attempt.get("shadow", False)
            and guess_attempt.get("correct") is False
        ):
            cumulative_failed += 1

        # Best possible score after this turn: max_turns - (turn_idx + 1) -
        # penalty*cumulative_failed
        # Which equals: max_turns - turn_num - penalty*cumulative_failed
        # Score guaranteed <= 0 when: penalty*cumulative_failed >= max_turns - turn_num
        if penalty * cumulative_failed >= max_turns - turn_num:
            return turn_num

    return None


class TeeWriter:
    """Write to both a file and stdout."""

    def __init__(self, file_path: Path) -> None:
        """Open the report output and initialize an in-memory copy."""
        self._resources = ExitStack()
        self.file = self._resources.enter_context(file_path.open("w"))
        self.buffer = io.StringIO()

    def write(self, text: str) -> None:
        """Write text to the report file, memory buffer, and standard output."""
        self.file.write(text)
        self.buffer.write(text)
        sys.stdout.write(text)

    def close(self) -> None:
        """Close the report output file and release owned resources."""
        self._resources.close()


def setup_matplotlib_style() -> None:
    """Configure consistent matplotlib style for all plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save figure with consistent settings."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {path}")
