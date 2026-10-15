"""
Logging utilities for the seizure detection system.
"""
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from ..config import LoggingConfig


def setup_logger(name: str, log_file: str = None, level: str = None) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file path (defaults to config)
        level: Optional log level (defaults to config)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    log_level = getattr(logging, (level or LoggingConfig.LOG_LEVEL).upper())
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_path = log_file or LoggingConfig.LOG_FILE
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=LoggingConfig.MAX_BYTES,
        backupCount=LoggingConfig.BACKUP_COUNT
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


# Create default logger
logger = setup_logger('seizure_detection')
