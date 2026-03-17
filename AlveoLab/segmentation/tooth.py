import numpy as np

from AlveoLab.utils import mask_or, LazyAttribute
from AlveoLab.trimesh_utils import get_oriented_bounding_box
import AlveoLab.math.geometry as geom


class Tooth:
    def __init__(self, area_groups, key, palmer=None):
        self.area_groups = area_groups
        self.key = key
        self.palmer = palmer
        self.mesh = area_groups[0].mesh

        self.mask = mask_or(*(s.mask for s in self.area_groups))
        self.orienter = None
        self.parent_orienter = self.area_groups[0].quadratic.orienter
        self.centre_of_mass = geom.center_of_mass(self.mesh.triangles_center[self.mask])
        self.quadratic = area_groups[0].quadratic

        seen = set()
        self.peaks = []
        for s in self.area_groups:
            for p in s.peaks:
                if id(p) not in seen:
                    seen.add(id(p))
                    self.peaks.append(p)

        root = self.quadratic.get_root_at(self.centre_of_mass)
        self.tangent, self.distal, self.buccal = [
            geom.normalize_vector(method(root=root)) for method in (
                self.quadratic.tangent_at,
                self.quadratic.distal_at,
                self.quadratic.buccal_at,
            )
        ]

    @LazyAttribute
    def area(self):
        return np.sum([g.area for g in self.area_groups])

    @LazyAttribute
    def obb(self):
        submesh = self.mesh.submesh([self.mask], append=True)
        return get_oriented_bounding_box(submesh)

