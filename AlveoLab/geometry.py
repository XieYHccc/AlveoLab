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
