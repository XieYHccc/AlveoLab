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


class _LazyAttribute(property):
    pass


def LazyAttribute(func):
    """

    :rtype: property
    """
    attr = func.__name__
    priv_attr = "_" + attr

    def getter(self):
        if not hasattr(self, priv_attr):
            setattr(self, priv_attr, func(self))
        return getattr(self, priv_attr)

    def deleter(self):
        if hasattr(self, priv_attr):
            delattr(self, priv_attr)

    return _LazyAttribute(getter, None, deleter, func.__doc__)