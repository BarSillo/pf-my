import logging
import time

# Store the start time when module is imported
START_TIME = time.time()

class ElapsedTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        """Add elapsed time since program start to the log message"""
        elapsed_seconds = record.created - START_TIME
        return f"{elapsed_seconds:.3f}s"

def setup_logger(name='rlhedge'):
    """Setup and return a logger with elapsed time formatting"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Only add handler if it doesn't exist
        handler = logging.StreamHandler()
        formatter = ElapsedTimeFormatter(
            fmt='[%(asctime)s] %(levelname)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    
    return logger
