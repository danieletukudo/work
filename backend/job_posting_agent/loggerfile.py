# logging 
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(logger_name: str) -> logging.Logger:
    """
    Configure logging with both file and console handlers
    Args:
        logger_name: Name of the logger
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_directory = "job_posting_agent/logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    # Create log file with timestamp
    log_file = os.path.join(log_directory, f"{logger_name}_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Test log messages
    logger.debug(f"Logger '{logger_name}' initialized. Logging to: {log_file}")
    
    return logger