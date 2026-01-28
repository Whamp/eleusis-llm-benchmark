"""Utility functions for analysis module."""

import io
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def compute_counting_cutoff(turns: list[dict], max_turns: int, penalty: int = 2) -> int | None:
    """Compute turn number where score becomes guaranteed <= 0.

    After turn n with k cumulative failed guesses, best possible score is:
        max_turns - (n+1) - penalty*k
    When penalty*k >= max_turns - n, score is guaranteed <= 0.

    Args:
        turns: List of turn dicts from results.json (with turn_number, guess_attempt)
        max_turns: Maximum turns allowed in the game
        penalty: Points lost per failed guess (default 2)

    Returns:
        Turn number where score became guaranteed <= 0, or None if never hit.
    """
    cumulative_failed = 0
    for turn in turns:
        turn_num = turn["turn_number"]  # 1-indexed
        turn_idx = turn_num - 1  # 0-indexed for scoring

        guess_attempt = turn.get("guess_attempt")
        if guess_attempt and not guess_attempt.get("shadow", False):
            if guess_attempt.get("correct") is False:
                cumulative_failed += 1

        # Best possible score after this turn: max_turns - (turn_idx + 1) - penalty*cumulative_failed
        # Which equals: max_turns - turn_num - penalty*cumulative_failed
        # Score guaranteed <= 0 when: penalty*cumulative_failed >= max_turns - turn_num
        if penalty * cumulative_failed >= max_turns - turn_num:
            return turn_num

    return None


class TeeWriter:
    """Write to both a file and stdout."""

    def __init__(self, file_path: Path):
        self.file = open(file_path, "w")
        self.buffer = io.StringIO()

    def write(self, text: str):
        self.file.write(text)
        self.buffer.write(text)
        print(text, end="")

    def close(self):
        self.file.close()


def setup_matplotlib_style():
    """Configure consistent matplotlib style for all plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150):
    """Save figure with consistent settings."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {path}")
