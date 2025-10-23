import numpy as np

import AlveoLab.math.geometry as geom
from AlveoLab.math.geometry import inner_product
from AlveoLab.trimesh_utils import get_oriented_bounding_box
from AlveoLab.utils import LazyAttribute

from trimesh.bounds import oriented_bounds


def get_common_width(min_a, max_a, min_b, max_b):
    """Imagine two parallel 1D lines a (min_a to max_a) and b (min_b to max_b).

    a               min_a |--------------| max_a
    b            min_b |--------| max_b
    common_width??        |-----|

    How much width do a and b have in common?
    If they don't overlap then the return value is negative the distance between them.
    a and b are interchangeable.
    """
    #print((min_a, max_a, min_b, max_b))

    # Cycle through all the cases
    if min_a <= max_a <= min_b <= max_b:
        # a     |----|
        # b             |----|
        return max_a - min_b

    if min_a <= min_b <= max_a <= max_b:
        # a      |---------|
        # b           |--------|
        return max_a - min_b

    if min_a <= min_b <= max_b <= max_a:
        # a   |---------------|
        # b       |-------|
        return max_b - min_b

    # Otherwise swap a and b and try again
    return get_common_width(min_b, max_b, min_a, max_a)


class Overlap1D(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.min_a = np.min(a)
        self.min_b = np.min(b)
        self.max_a = np.max(a)
        self.max_b = np.max(b)

        self.width_a = self.max_a - self.min_a
        self.width_b = self.max_b - self.min_b

        self.width = get_common_width(self.min_a, self.max_a, self.min_b, self.max_b)

        self.ratio = self.width / min(self.width_b, self.width_a)


class OverlappingAreaGroup:
    def __init__(self, peaks, mask, mesh, orienter, quadratic):
        self.peaks = peaks
        self.mask = mask
        self.mesh = mesh
        self.orienter = orienter
        self.peak_points = np.array([mesh.vertices[peak] for peak in self.peaks])
        self.quadratic = quadratic
        self.points = self.mesh.triangles_center[self.mask]
        self.centre_of_mass = geom.center_of_mass(self.points)

        self.update_quadratic(quadratic)
        # root = quadratic.get_root_at(self.centre_of_mass)
        # self.tangent, self.distal, self.buccal = [
        #     geom.normalize_vector(method(root=root)) for method in (
        #         quadratic.tangent_at,
        #         quadratic.distal_at,
        #         quadratic.buccal_at,
        #     )
        # ]
        #
        # a = geom.inner_product(self.points, self.tangent)
        # self.span = self.points[[np.argmin(a), np.argmax(a)]]
        #
        # heights = geom.inner_product(self.peak_points, orienter.occlusal)
        #
        # self.min_height = np.min(heights)
        # self.max_height = np.max(heights)

    def __repr__(self):
        return "{}(peak_indices={})".format(self.__class__.__name__, self.peaks)

    @LazyAttribute
    def obb(self):
        submesh = self.mesh.submesh([self.mask], append=True)
        return get_oriented_bounding_box(submesh)

    @LazyAttribute
    def width(self):
        # xs, ys = self.orienter.to_horizontal(self.span)
        # return self.quadratic.quadratic_2d.get_distance_between_points(*xs, *ys)
        return geom.magnitude(self.span[1] - self.span[0])

    @LazyAttribute
    def is_one_axis_dominate(self):
        ratios = self.obb.extents / np.max(self.obb.extents)
        dominate_count = np.sum(ratios < 0.3)
        return dominate_count >= 2

    @LazyAttribute
    def area(self):
        return np.sum(self.mesh.area_faces[self.mask])

    @LazyAttribute
    def is_one_sided(self):
        """Attempts to determine if this area is solely on either the lingual or buccal side,
        not both. This is nowhere near as strong as I'd like it to be.

        Currently it is done by:
        1. Looking at the triangle unit normals in this area.
        2. Discard those that are too parallel to the jaw-line.
        3. Transform the remaining normals to 2D with buccal and occlusal as the new axes.
        4. The groups on the rugae will all face only palatally so will be rejected by this rule.
        5. Convert the 2D vectors to angles.
        6. Look at the spread of those angles.
        """

        # Steps 1 and 2
        mask = self.mask.copy()
        units = self.mesh.face_normals[mask]
        units = units[np.abs(inner_product(units, self.distal)) < 0.5]

        if len(units) == 0:
            return True

        # Step 3
        dx, dy = geom.get_components(units, self.buccal, self.orienter.occlusal)

        # Step 4
        num_buccal_face = dx[dx > 0.4].shape[0]
        num_lingual_face = dx[dx < -0.6].shape[0]
        buccal_ratio = num_buccal_face / units.shape[0]
        lingual_ratio = num_lingual_face / units.shape[0]
        if buccal_ratio < 0.05 or (1 - lingual_ratio - buccal_ratio) > 0.9:
            return True

        # Step 5
        thetas = np.arctan2(dy, dx)

        # Step 6
        # The thetas are far too noisy to get anything meaningful with min and max values. It needs
        # to take into account the density of thetas. It's visually obvious when you plot them but
        # not to the computer. After getting nowhere with np.histogram I eventually settled with
        # just this.
        std = np.std(thetas)
        return std < 0.5

    def get_inline_overlap(self, other):

        both = (self, other)

        # Combine each tangent to the quadratic to get a single overall mean tangent.
        tangent = geom.normalize_vector(sum(i.tangent for i in both))

        # inner product the tangent vector with all points from each area group
        # to get two 1D arrays of projections parallel to the jaw line.
        projections = (geom.inner_product(i.span, tangent) for i in both)

        # Get the 1d overlap of the two arrays.
        return Overlap1D(*projections)

    def update_quadratic(self, quadratic):
        self.quadratic = quadratic

        root = quadratic.get_root_at(self.centre_of_mass)
        self.tangent, self.distal, self.buccal = [
            geom.normalize_vector(method(root=root)) for method in (
                quadratic.tangent_at,
                quadratic.distal_at,
                quadratic.buccal_at,
            )
        ]

        a = geom.inner_product(self.points, self.tangent)
        self.span = self.points[[np.argmin(a), np.argmax(a)]]

        heights = geom.inner_product(self.peak_points, self.orienter.occlusal)

        self.min_height = np.min(heights)
        self.max_height = np.max(heights)

        del self.width
