"""Demo script to showcase colored logging."""

import logging

from eleusis.logging_utils import setup_logging

# Setup logging
setup_logging(
    log_file="logs/demo_log.txt",
    console_level=logging.DEBUG,  # Show all levels in demo
    file_level=logging.DEBUG,
)

logger = logging.getLogger(__name__)

# Test all log levels
logger.debug("This is a DEBUG message (gray)")
logger.info("This is an INFO message (white)")
logger.warning("This is a WARNING message (yellow)")
logger.error("This is an ERROR message (red)")

print("\nCheck logs/demo_log.txt to see the detailed file output with timestamps!")
