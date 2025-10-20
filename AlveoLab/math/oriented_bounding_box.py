import numpy as np

from AlveoLab.utils import LazyAttribute
class Obb:
    def __init__(self, to_origin_matrix, extents):
        self.to_origin_matrix = to_origin_matrix
        self.extents = extents

    @LazyAttribute
    def center(self):
        inv = np.linalg.inv(self.to_origin_matrix)
        center_homogeneous = inv @ np.array([0, 0, 0, 1])
        return center_homogeneous[:3]