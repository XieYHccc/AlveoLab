import numpy as np

from AlveoLab.utils import LazyAttribute


class Obb:
    def __init__(self, to_origin_matrix, extents):
        self.to_origin_matrix = to_origin_matrix  # axis correspond to extents
        self.extents = extents  # from min to max

    @LazyAttribute
    def center(self):
        to_origin_offset = self.to_origin_matrix[:3, 3]
        return self.to_origin_matrix[:3, :3].T @ -to_origin_offset

    @LazyAttribute
    def max_width(self):
        return self.extents[2]

