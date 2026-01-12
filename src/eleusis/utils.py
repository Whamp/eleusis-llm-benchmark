"""Utility functions shared across the Eleusis codebase."""

import logging
import re
import sys
from pathlib import Path

__all__ = ["model_spec_to_display_name", "ColoredFormatter", "setup_logging"]


def model_spec_to_display_name(model_spec: str) -> str:
    """Convert model spec to readable display name.

    Examples:
        "openrouter:anthropic/claude-3.5-haiku" -> "Claude 3.5 Haiku"
        "hf:meta-llama/Llama-3.3-70B" -> "Llama 3.3 70b"
    """
    if ":" in model_spec:
        _, model_name = model_spec.split(":", 1)
    else:
        model_name = model_spec

    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    model_name = model_name.replace("-", " ").replace("_", " ")
    model_name = re.sub(r'\s+', ' ', model_name).strip()
    return model_name.title()


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log messages for terminal output."""

    COLORS = {
        "DEBUG": "\033[90m",
        "INFO": "\033[97m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[91m\033[1m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color codes."""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        record.msg = f"{log_color}{record.msg}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_file: str | Path,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    file_format: str = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    console_format: str = "%(message)s",
    colored: bool = True,
) -> None:
    """Setup dual logging: colored console + detailed file output."""
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(file_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)

    if colored:
        console_handler.setFormatter(ColoredFormatter(console_format))
    else:
        console_handler.setFormatter(logging.Formatter(console_format))

    logging.basicConfig(
        level=min(console_level, file_level),
        handlers=[file_handler, console_handler],
        force=True,
    )
