import numpy as np
import queue
import collections

import AlveoLab.math.geometry as geom
from AlveoLab.math.geometry import inner_product
from AlveoLab.utils import LazyAttribute, mask_or
from AlveoLab.trimesh_utils import get_edge_based_curvature, get_face_face_adjacency
from AlveoLab.orienter.pca_dental_orienter import PcaOrienter
from AlveoLab.math.least_square_quadratic import Quadratic3D
from AlveoLab.segmentation.overlapping_area_group import OverlappingAreaGroup
from AlveoLab.segmentation.tooth import Tooth
from AlveoLab.grouping import Grouping


class CurvatureBasedSeg:
    _MAX_COST = 1.65
    _MAX_SPREAD_WIDTH = 10
    _MAX_SPREAD_HEIGHT = 12

    # 10 year teeth
    MAX_TOOTH_WIDTH = 13
    MAX_PEAK_DISTANCE = 15
    MIN_TOOTH_AREA = 15
    MAX_TOOTH_HEIGHT = 15

    @LazyAttribute
    def tri2tri_costs(self):
        """The costs, based on curvature, of moving from each triangle to each of it's adjacent
        triangles. It is an array of shape (number of triangles, 3). Triangles are referenced by
        argument based on the order they are listed in the original mesh.
        """
        # We are only looking for the crease where tooth meets gum. Creases / slots / grooves are
        # represented with a negative sign in `mesh.curvature.signed` whereas bumps have positive
        # sign. .clip(max=0) sets all positive values to 0.
        creases_only = -self.edge_curvature_face_view .clip(max=0)

        # An L2 norm seemed to work well, hence the square.
        # return np.ascontiguousarray((creases_only ** 2).clip(max=self._MAX_COST * 1.1))
        return creases_only.clip(max=self._MAX_COST * 1.1)

    @property
    def discarded_peaks_all(self):
        """Contains all the peak args that we don't want. It comes from flattening
        `self.discarded_peaks`. Any group that contains any of these should be removed. """

        discarded_peaks_all = []
        for teeth_list in self.discarded_teeth.values():
            for tooth in teeth_list:
                discarded_peaks_all.append(tooth.peaks)

        for group_list in self.discarded_overlap_groups.values():
            for group in group_list:
                discarded_peaks_all.append(group.peaks)

        for v in self.discarded_peaks.values():
            discarded_peaks_all.append(v)

        if len(discarded_peaks_all) == 0:
            return set()
        return set.union(*discarded_peaks_all)

        # return set.union(*self.discarded_peaks.values())

    @property
    def valid_peaks(self):
        filtered_peaks = np.array(
            [peak for peak in self._peak_indices if peak not in self.discarded_peaks_all])
        return filtered_peaks

    # def plot_curvature_hist(self):
    #     plt.hist(self.edge_curvature, bins=30)
    #     plt.show()
    #
    # def plot_peak_spread_region(self, peak_idx):
    #     self._mesh.visual.face_colors[self.peak_masks[peak_idx]] = trimesh.visual.random_color()
    #
    # def plot_spread_regions(self):
    #     # mask = np.array([value for value in self.peak_masks.values()])
    #     # mask = np.bitwise_or.reduce(mask, axis=0)
    #     # self._mesh.visual.face_colors[mask] = [255, 0, 0, 255]
    #     for peak in self._peak_indices:
    #         self.plot_peak_spread_region(peak)

    # def plot_group_regions(self):
    #     for group in self._peak_groups:
    #         mask = [self.peak_masks[peak_] for peak_ in self._peak_indices[group]]
    #         mask = np.bitwise_or.reduce(mask, axis=0)
    #         self._mesh.visual.face_colors[mask] = trimesh.visual.random_color()

    def __init__(self, mesh, orienter: PcaOrienter, peaks_idx):
        self._mesh = mesh
        self._orienter = orienter
        self._peak_indices = peaks_idx

        self.faces_adj = get_face_face_adjacency(self._mesh)
        self.edge_curvature, self.edge_curvature_face_view = get_edge_based_curvature(self._mesh, get_map=True)

        self.peak_masks = {}  # peak_id : triangle_mask
        self.peak_costs = {}  # peak_id : accumulative cost to each triangle
        self.overlapping_area_args = []  # list of sets of peak args
        self.overlapping_area_groups = []
        self.inline_group_args = []
        self.teeth = []

        self.discarded_peaks = collections.defaultdict(set)
        self.discarded_overlap_groups = collections.defaultdict(list)  # peaks that are considered as rugae
        self.discarded_teeth = collections.defaultdict(list)
        self.quadratic = None

        self._run()

    def _run(self):
        self._spread_from_peaks()
        self._build_quadratic()
        self._build_overlapping_area_groups()
        self._build_quadratic()
        for group in self.overlapping_area_groups:
            group.update_quadratic(self.quadratic)
        # self._remove_peaks_on_rugae()

        self._group_inline_area_groups()
        self._build_teeth()

    def _spread_from_peaks(self):
        for peak in self._peak_indices:
            self._spread_from_peak(peak, self.tri2tri_costs)

    def _spread_from_peak(self, peak_idx, costs):  # core region growing algorithm
        peak_triangles = np.where(self._mesh.faces == peak_idx)[0]

        # init accumulative costs
        accumulative_cost = np.ones(self._mesh.faces.shape[0]) * self._MAX_COST
        accumulative_cost[peak_triangles] = 0.0

        # init queue and shortest flags
        is_shortest = np.zeros(self._mesh.faces.shape[0], dtype=bool)  # 1 means the face's minimum accumulative cost
        faces_init = self.faces_adj[peak_triangles].reshape(-1)  # has been got
        que = queue.PriorityQueue()
        [que.put((accumulative_cost[face], face)) for face in faces_init]

        # spread from triangles containing peak
        while not que.empty():
            face = que.get()[1]
            if is_shortest[face] == 1:
                continue
            else:
                is_shortest[face] = 1

                # make sure the region won't spread too widely
                face_center = self._mesh.triangles_center[face]
                width1 = abs(np.inner(self._orienter.right, face_center - self._mesh.vertices[peak_idx]))
                width2 = abs(np.inner(self._orienter.forward, face_center - self._mesh.vertices[peak_idx]))
                height = abs(np.inner(self._orienter.occlusal, face_center - self._mesh.vertices[peak_idx]))
                if width1 > self._MAX_SPREAD_WIDTH or width2 > self._MAX_SPREAD_WIDTH or height > self._MAX_SPREAD_HEIGHT:
                    # this peak beyond the max spreading range, discard it
                    self.discarded_peaks['Spilled Peaks'].add(peak_idx)
                    return
                    # continue

                face_adj = self.faces_adj[face]
                edges_cost_adj = costs[face]
                for i, face_ in enumerate(face_adj):
                    if accumulative_cost[face] + edges_cost_adj[i] < accumulative_cost[face_]:
                        accumulative_cost[face_] = accumulative_cost[face] + edges_cost_adj[i]
                        que.put((accumulative_cost[face_], face_))

        self.peak_costs[peak_idx] = accumulative_cost
        self.peak_masks[peak_idx] = is_shortest

    # def _remove_peaks_on_rugae(self):
    #     """
    #     Any tooth should have both a lingual and a buccal side, or for very
    #     slanted teeth, at least a significant variance. The groups on the rugae
    #     will all face only palatally so will be rejected by this rule.
    #     """
    #     # 1. fit a quadratic curve to all spread regions
    #     all_region_mask = np.zeros_like(self.peak_masks[self.valid_peaks[0]], dtype=bool)
    #
    #     for p in self.valid_peaks:
    #         all_region_mask |= self.peak_masks[p]
    #     triangle_centers = self._mesh.triangles_center[all_region_mask]
    #     x, y = (np.dot((triangle_centers - self._orienter.center), e)
    #             for e in (self._orienter.right, self._orienter.forward))
    #
    #     # prioritise the more occlusal points
    #     weights = np.dot(triangle_centers, self._orienter.occlusal)
    #     weights -= np.min(weights)
    #     weights = weights ** 5
    #
    #     poly = np.polynomial.Polynomial.fit(x, y, 2, w=weights)
    #     deriv = poly.deriv()
    #
    #     # 2. check each group's region
    #     groups_to_remove = []
    #     for group, mask in self._group_region_masks.items():
    #         center = self._mesh.triangles_center[mask].mean(axis=0)
    #         deriv_at_peak = deriv((center - self._orienter.center) @ self._orienter.right)
    #         # tangent in right–forward plane
    #         tangent = geom.normalize_vector(
    #             deriv_at_peak * self._orienter.forward + self._orienter.right
    #         )
    #         # normal (approx lingual) direction
    #         approx_lingual_dir = geom.normalize_vector(np.cross(tangent, self._orienter.occlusal))
    #         region_faces = np.where(mask)[0]
    #         region_normals = self._mesh.face_normals[region_faces]
    #         dot = np.dot(region_normals, approx_lingual_dir)
    #         num_buccal_face = dot[dot < -0.7].shape[0]
    #         num_lingual_face = dot[dot > 0.7].shape[0]
    #         buccal_ratio = num_buccal_face / region_faces.shape[0]
    #         lingual_ratio = num_lingual_face / region_faces.shape[0]
    #         if buccal_ratio < 0.05 or (1 - lingual_ratio - buccal_ratio) > 0.9:
    #             self.discarded_overlap_groups[group] = mask
    #             groups_to_remove.append(group)
    #     for group in groups_to_remove:
    #         self._group_region_masks.pop(group)
    #
    #     # flatten peaks
    #     peaks_to_remove = set().union(*groups_to_remove)
    #     self.discarded_peaks['Rugae Peaks'] = peaks_to_remove

    def _build_overlapping_area_groups(self):
        """
        Find all peak spreads that share area on the mesh. The peak groups are stored in
        `self.overlapping_arg_groups`, an array of sets of args.

        It allows indirect groups. i.e. If peaks[0] overlaps with peaks[1] and peaks[1] overlaps
        with peaks[2] but peaks[0] doesn't overlap with peaks[2] then they are all grouped
        together anyway.
        """
        filtered_peaks = self.valid_peaks
        num_peak = len(filtered_peaks)

        # `overlap_adjacency_matrix` is a square bool array.
        # `overlap_adjacency_matrix[i, j]` = do peaks[i] and peaks[j] overlap?
        overlap_adjacency_matrix = np.zeros((num_peak, num_peak))
        for i in range(num_peak):
            overlap_adjacency_matrix[i][i] = 1
            mask1 = self.peak_masks[filtered_peaks[i]]
            for j in range(i + 1, num_peak):
                mask2 = self.peak_masks[filtered_peaks[j]]
                if np.bitwise_and(mask1, mask2).any():
                    overlap_adjacency_matrix[i][j] = 1
                    overlap_adjacency_matrix[j][i] = 1

        # build overlapping area groups
        flag = np.zeros(num_peak, dtype=bool)
        for i in range(num_peak):
            if flag[i]:
                continue
            connected = np.where(overlap_adjacency_matrix[i] == 1)[0]
            group_peak_args = np.unique(np.where(overlap_adjacency_matrix[connected] == 1)[1])
            flag[group_peak_args] = True
            group_peaks = set(filtered_peaks[group_peak_args])
            mask = mask_or(*(self.peak_masks[i] for i in group_peaks))
            self.overlapping_area_args.append(group_peaks)
            area_group = OverlappingAreaGroup(group_peaks, mask, self._mesh, self._orienter, self.quadratic)

            if area_group.width > self.MAX_TOOTH_WIDTH:
                # Occasionally you get very long stretches of gum just beneath
                # the incisors.
                self.discarded_overlap_groups["Too Wide"].append(area_group)
                # self.overlapping_area_groups.append(area_group)
            elif area_group.is_one_axis_dominate:
                # To be a tooth the area should have significant extent
                # in at least two axes.
                self.discarded_overlap_groups["One Axis Dominate"].append(area_group)
                # self.overlapping_area_groups.append(area_group)
            elif area_group.is_one_sided:
                # To be a cusp of a tooth the area should have both lingual
                # facing and buccal facing parts.
                self.discarded_overlap_groups["Only on One Side"].append(area_group)
                # self.overlapping_area_groups.append(area_group)
            else:
                self.overlapping_area_groups.append(area_group)

    def _build_quadratic(self):
        """Build the quadratic (approximation of the jaw line) fitting to the point of each peak
        that isn't `spilled`. Use the quadratic to sort and enumerate the peaks (including the
        spilled ones) by their position along the quadratic. Modify `self.peaks` and
        `self.peak_points` to reflect the reordering.
        """

        peak_points_unspilled = np.array(self._mesh.vertices[self.valid_peaks])

        assert len(peak_points_unspilled) >= 3, "Not enough valid peaks to build quadratic"
        self.quadratic = Quadratic3D(peak_points_unspilled, self._orienter)

        # This just tests "how tall is the quadratic?".
        ys = self.quadratic.quadratic_2d.points[:, 1]
        assert self.quadratic.quadratic_2d.height > 1.0 * ys.std()
        """Least squares quadratic is a poor approximation of the jaw line. This
        typically happens if there are raised areas in the centre-rear of the
        model. Other than manually removing these areas, there is nothing that
        can be done to fix this."""

    def _group_inline_area_groups(self):
        """Next group overlapping_area_groups if they are inline to join the lingual and buccal
        cusps of molars/premolars. This is done by projecting the points in each area quadratic
        to get a 1D line of points. The range of each area's points is found and compared with
        the ranges from other areas to determine if they are inline. """

        # Overlap in this method refers to the ranges of the projections overlapping rather than
        # areas overlapping as it was before. Apart from `self.overlapping_area_groups`.

        # This method considers both the absolute width of a range overlap (mm) and the ratio of
        # overlap_width / min(width of each range).

        n = len(self.overlapping_area_groups)
        assert n > 0

        overlap_width_map = np.zeros((n, n))
        overlap_ratio_map = np.zeros((n, n))

        # Cycle through all possible pairs, recording the width and ratios in
        # the above square arrays.
        for i in range(n):
            area_group_i = self.overlapping_area_groups[i]
            for j in range(i + 1, n):
                area_group_j = self.overlapping_area_groups[j]

                # Skip if they are more than a tooth's width apart.
                # This is approximated lazily by looking at the last and first peak of each group
                peak_point1 = self._mesh.vertices[max(area_group_i.peaks)]
                peak_point2 = self._mesh.vertices[min(area_group_j.peaks)]
                if geom.magnitude(peak_point1 - peak_point2) > self.MAX_TOOTH_WIDTH:
                    continue

                # Skip if they are too far apart in the occlusal direction.
                if abs(inner_product(area_group_i.obb.center - area_group_j.obb.center, self._orienter.occlusal)) > 3:
                    continue

                # The actual maths is handled in `OverlappingAreasGroup.get_inline_overlap`
                overlap = area_group_i.get_inline_overlap(area_group_j)
                overlap_width_map[i, j] = overlap.width
                overlap_ratio_map[i, j] = overlap.ratio

        # Magic made up rule that combines all the above into a hard "inline or not inline" square
        # bool array.
        mask = ((12 > overlap_width_map) & (overlap_width_map > 2) & (overlap_ratio_map > 0.55))

        # Again convert bool array to arg groups
        self.inline_group_args = Grouping(mask).groups

    def _build_teeth(self):
        """Convert each group from `self.inline_group_args` to a Tooth instance from tooth_class.py.
        Also filters away instances that cover too little area to be a tooth. Otherwise you get tiny
        little isolated bumps which are irrelevant. """

        self.teeth = []
        for (i, args) in enumerate(self.inline_group_args):
            # Each tooth receives all the OverlappingAreaGroup objects from an inline group. We
            # don't know which tooth is which yet so each is given an enumeration as a convenient ID.
            groups = [self.overlapping_area_groups[j] for j in sorted(args)]
            tooth = Tooth(groups, i)
            if tooth.area < self.MIN_TOOTH_AREA:
                self.discarded_teeth["Area too small"].append(tooth)
                continue

            self.teeth.append(tooth)



        









