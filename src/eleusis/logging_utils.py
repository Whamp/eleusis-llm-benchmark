"""Reusable colored logging utilities with dual output (console + file)."""

import logging
import sys
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log messages for terminal output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[90m",  # Gray
        "INFO": "\033[97m",  # Bright white
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m\033[1m",  # Bold red
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
    """Setup dual logging: colored console + detailed file output.

    Args:
        log_file: Path to log file
        console_level: Logging level for console (default: INFO)
        file_level: Logging level for file (default: DEBUG)
        file_format: Format string for file output (default: includes timestamp, file, line)
        console_format: Format string for console output (default: message only)
        colored: Whether to use colored output on console (default: True)
    """
    # Create file handler with detailed formatting
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(file_format))

    # Create console handler with optional colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)

    if colored:
        console_handler.setFormatter(ColoredFormatter(console_format))
    else:
        console_handler.setFormatter(logging.Formatter(console_format))

    # Configure root logger
    logging.basicConfig(
        level=min(console_level, file_level),  # Capture everything at lowest level
        handlers=[file_handler, console_handler],
        force=True,  # Override any existing configuration
    )
