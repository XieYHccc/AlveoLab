import numpy as np


def normalize_vector(vec):
    """
    Normalize a vector to have a length of 1.

    Parameters:
    --------------
    vec: (n, ) float
        Input vector to be normalized.

    Returns:
    -----------
    numpy.ndarray: (n, ) float
        Normalized vector with length 1, or the original vector if it is a zero vector.
    """

    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec  # 如果是零向量，直接返回原始向量
    return vec / norm


