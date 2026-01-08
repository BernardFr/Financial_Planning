import logging
import os
LOG_LEVEL = 'INFO'
LOG_NAME = 'Financial_Plan'

logger = logging.getLogger(LOG_NAME)
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    # Create log file name by replacing .py with .log
    log_filename = os.path.splitext(os.path.basename(__file__))[0] + ".log"
    
    # File handler for INFO and above (saves to log file)
    fh = logging.FileHandler(log_filename)
    fh.setLevel(LOG_LEVEL)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    
    # Console handler (displays on terminal)
    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    # Optional: Prevent log messages from propagating to the root logger
    logger.propagate = False


