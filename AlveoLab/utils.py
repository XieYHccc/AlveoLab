import logging
import time
import types

import numpy as np

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

def copy_name_wrapper(wrapper):
    def wrapped(function):
        wrapped_function = wrapper(function)
        wrapped_function.__name__ = function.__name__
        wrapped_function.__qualname__ = function.__qualname__
        wrapped_function.__doc__ = function.__doc__
        return wrapped_function

    return wrapped

@copy_name_wrapper
def accept_generators(function):
    def wrapped(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], types.GeneratorType):
            return function(*args[0], **kwargs)
        return function(*args, **kwargs)

    return wrapped

@accept_generators
def mask_or(mask, *masks):
    for m in masks:
        mask = mask | m
    return mask

def sep_last_ax(points):
    points = np.asarray(points)
    return tuple(points[..., i] for i in range(points.shape[-1]))


def zip_axes(*axes):
    """Convert vertex data from separate arrays for x, y, z to a single
    combined points array like most vpl functions require.

    :param axes: Each separate axis to combine.
    :type axes: numpy.ndarray

    All **axes** must have the matching or broadcastable shapes. The number of
    axes doesn't have to be 3.

    .. code-block:: python

        import vtkplotlib as vpl
        import numpy as np

        vpl.zip_axes(np.arange(10),
                     4,
                     np.arange(-5, 5))

        # Out: array([[ 0,  4, -5],
        #             [ 1,  4, -4],
        #             [ 2,  4, -3],
        #             [ 3,  4, -2],
        #             [ 4,  4, -1],
        #             [ 5,  4,  0],
        #             [ 6,  4,  1],
        #             [ 7,  4,  2],
        #             [ 8,  4,  3],
        #             [ 9,  4,  4]])

    .. seealso:: `unzip_axes()` for the reverse.

    """

    return np.concatenate(
        [i[..., np.newaxis] for i in np.broadcast_arrays(*axes)], axis=-1)


def unzip_axes(points):
    """Separate each component from an array of points.

    :param points: Some points.
    :type points: numpy.ndarray

    :return: Each axis separately as a tuple.
    :rtype: tuple

    See `zip_axes()` more information and the reverse.

    """

    return sep_last_ax(points)

def copy_name_wrapper(wrapper):
    def wrapped(function):
        wrapped_function = wrapper(function)
        wrapped_function.__name__ = function.__name__
        wrapped_function.__qualname__ = function.__qualname__
        wrapped_function.__doc__ = function.__doc__
        return wrapped_function

    return wrapped

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

