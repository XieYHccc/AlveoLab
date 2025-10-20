import numpy as np


def normalize_vector(vec):
    """
    Normalize a vector to have a length of 1.

    Parameters:
    ----------
    vec: (n, ) float
        Input vector to be normalized.

    Returns:
    ----------
    numpy.ndarray: (n, ) float
        Normalized vector with length 1, or the original vector if it is a zero vector.
    """

    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec  # 如果是零向量，直接返回原始向量
    return vec / norm


def center_of_mass(points, weights=None):
    """
    The (weighted) mean of **points**.

    Parameters:
    ----------
    points: (n, m) float
        The points to find the center of mass of.
    weights: (n, ) float, optional

    Returns:
    ----------
    np.ndarray: (m, ) float
        The center of mass of the points.
    """
    if weights is None:
        return np.array([i.mean() for i in points.T])
    else:
        weights = weights[(...,) + (np.newaxis,) * (points.ndim - weights.ndim)]
        return np.array([i.sum() for i in (points * weights).T]) / weights.sum()


def np_sum_m1(x, keepdims=False):
    """Performs ``np.sum(x, axis=-1, keepdims=keepdims)`` but faster."""
    return reduce_last_ax(x, np.add, keepdims)


def reduce_last_ax(x, operator, keepdims=False):
    if x.shape[-1] == 0:
        return x

    itr = iter(x.T)
    out = next(itr)
    for i in itr:
        out = operator(out, i)
    out = out.T

    if keepdims:
        out = out[..., np.newaxis]
    return out


def magnitude_sqr(vector, keepdims=False):
    """Calculates the square of the hypotenuse of a **vector**. This is faster
    than :meth:`magnitude` as it skips the square root and can be used to
    compare or sort distances.
    """
    return np_sum_m1(vector * vector, keepdims)


def magnitude(vector, keepdims=False):
    """Calculates the hypotenuse, magnitude or length  of a **vector** using
    Pythagoras.
    """
    return np.sqrt(magnitude_sqr(vector, keepdims))

def inner_product(a, b, keepdims=False):
    """Calculates the scalar/inner/interior/dot/"whatever you want to call it"
    product of |vectors| **a** and **b**, returning a scalar.

    Arguments **a** and **b** must be numpy-broadcastable.

    .. seealso::

        The :class:`UnitVector` class for a more convenient way to perform
        multiple :meth:`inner_product` calls.
    """
    return np_sum_m1(a * b, keepdims=keepdims)

def real_and_bounded(x, lb=0, ub=1):
    mask = x.imag == 0
    mask &= x.real >= lb
    mask &= x.real <= ub
    return x.real[mask]

def stagger(points, depth=1):
    ijs = [(i, len(points) - depth + i) for i in range(depth + 1)]
    return tuple(points[i:j] for (i, j) in ijs)

def get_components(points, *unit_vectors):
    return tuple(inner_product(points, uv) for uv in unit_vectors)

def get_components_zipped(points, *unit_vectors):
    return points @ np.array(unit_vectors).T
