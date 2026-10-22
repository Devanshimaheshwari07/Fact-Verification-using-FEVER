"""Centralized logging configuration for the FEVER IR system."""
import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = "fever_ir",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configures and returns a logger instance with console and optional file handlers.

    Args:
        name: Logger name.
        log_file: Optional relative or absolute path to log file.
        level: Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
