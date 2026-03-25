"""
Logging Configuration Module
==============================
Provides a configured logger with file and console handlers.
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str = "affordance", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure logger with file + console handlers.

    Args:
        name: Logger name (typically __name__ of the calling module).
        log_dir: Directory where log files are written.
        level: Logging level for the console handler.

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f"run_{timestamp}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d — %(message)s"
    ))
    logger.addHandler(fh)

    return logger
