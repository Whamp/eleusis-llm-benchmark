"""Utility functions for analysis module."""

import io
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


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
