import logging
import os
from datetime import datetime

def setup_logger(name="OCR", log_dir="logs", level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a daily log file
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"ocr_system_{date_str}.log")

    # Formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')

    # Handler 1: File
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)

    # Handler 2: Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate logs if reloading
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger