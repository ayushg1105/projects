"""
Structured logging module for the ColorPulse Vision platform.
Provides consistent formatting and level filtering across core engines, GUI, and services.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "ColorPulseVision", log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance with formatted console and optional file handlers.
    
    Args:
        name: Name of the logger component.
        log_file: Optional path to output log file.
        level: Minimum log level (default: logging.INFO).
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default application logger
app_logger = setup_logger()
