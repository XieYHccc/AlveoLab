import logging
import time

now = time.time


def get_logger(name=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # add console handler if not exist
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        log_format = logging.Formatter('[AlveoLab][%(levelname)s][%(name)s]: %(message)s')
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    return logger
