import time
import numpy as np
import queue
import trimesh

from AlveoLab.trimesh_utils import get_edge_based_curvature, get_face_face_adjacency
import matplotlib.pyplot as plt
from AlveoLab.orienter.pca_orienter import PcaOrienter

class CurvatureBasedSeg:

    _mesh: trimesh.Trimesh
    _orienter: PcaOrienter
    _curvature: np.ndarray
    _curvature_per_triangle: np.ndarray
    _peaks_idx: np.ndarray           # the indices of peaks
    _peak_masks: {}                  # peak_id : triangle_mask
    _peak_costs: {}                  # peak_id : accumulative cost to each triangle
    _peak_groups: []                 # each element is a ndarray containing peaks whose spreading region are overlap
    _overlap_mask: np.ndarray        # overlap_mask[i, j] = do peaks[i] and peaks[j] overlap?

    _MAX_COST = 2.0
    _MAX_SPREAD_WIDTH = 6
    _MAX_SPREAD_HEIGHT = 5

    @property
    def curvature(self):
        return self._curvature

    @property
    def curvature_per_triangle(self):
        return self._curvature_per_triangle

    @property
    def mask(self):
        return self._peak_masks

    def plot_curvature_hist(self):
        plt.hist(self._curvature, bins=30)
        plt.show()

    def plot_peak_spread_region(self, peak_idx):
        self._mesh.visual.face_colors[self._peak_masks[peak_idx]] = trimesh.visual.random_color()

    def plot_spread_regions(self):
        # mask = np.array([value for value in self._peak_masks.values()])
        # mask = np.bitwise_or.reduce(mask, axis=0)
        # self._mesh.visual.face_colors[mask] = [255, 0, 0, 255]
        for peak in self._peaks_idx:
            self.plot_peak_spread_region(peak)

    def plot_group_regions(self):
        for group in self._peak_groups:
            mask = [self._peak_masks[peak_] for peak_ in self._peaks_idx[group]]
            mask = np.bitwise_or.reduce(mask, axis=0)
            self._mesh.visual.face_colors[mask] = trimesh.visual.random_color()

    def __init__(self, mesh, orienter, peaks_idx):
        self._mesh = mesh
        self._orienter = orienter
        self._peaks_idx = peaks_idx
        self._faces_adj = get_face_face_adjacency(self._mesh)
        self._peak_masks = {}
        self._peak_costs = {}
        self._peak_groups = []

        t0 = time.time()
        self._run()
        run_time = time.time() - t0
        print(run_time)

    def _run(self):
        self._calculate_curvature()
        self._spread_from_peaks()
        self._group_overlap_region()

    def _calculate_curvature(self):
        self._curvature, self._curvature_per_triangle = \
            get_edge_based_curvature(self._mesh, get_map=True)

    def _spread_from_peaks(self):

        tri2tri_costs = -self._curvature_per_triangle.clip(max=0)

        # self._spread_from_peak(self._peaks_idx[2], tri2tri_costs)
        for peak in self._peaks_idx:
            self._spread_from_peak(peak, tri2tri_costs)

    def _spread_from_peak(self, peak_idx, costs):  # core region growing algorithm
        peak_triangles = np.where(self._mesh.faces == peak_idx)[0]

        # init accumulative costs
        accumulative_cost = np.ones(self._mesh.faces.shape[0]) * self._MAX_COST
        accumulative_cost[peak_triangles] = 0.0

        # init queue and shortest flags
        is_shortest = np.zeros(self._mesh.faces.shape[0], dtype=bool)  # 1 means the face's minimum accumulative cost
        faces_init = self._faces_adj[peak_triangles].reshape(-1)       # has been got
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
                width = abs(np.inner(self._orienter.right, face_center - self._mesh.vertices[peak_idx]))
                height = abs(np.inner(self._orienter.occlusal, face_center - self._mesh.vertices[peak_idx]))
                if width > self._MAX_SPREAD_WIDTH or height > self._MAX_SPREAD_HEIGHT:
                    continue

                face_adj = self._faces_adj[face]
                edges_cost_adj = costs[face]
                for i, face_ in enumerate(face_adj):
                    if accumulative_cost[face]+edges_cost_adj[i] < accumulative_cost[face_]:
                        accumulative_cost[face_] = accumulative_cost[face]+edges_cost_adj[i]
                        que.put((accumulative_cost[face_], face_))

        self._peak_costs[peak_idx] = accumulative_cost
        self._peak_masks[peak_idx] = is_shortest

    def _group_overlap_region(self):
        """
        Find all peak spreads that share area on the mesh. The peak groups are stored in
        `self.overlapping_arg_groups`, an array of sets of args.

        It allows indirect groups. i.e. If peaks[0] overlaps with peaks[1] and peaks[1] overlaps
        with peaks[2] but peaks[0] doesn't overlap with peaks[2] then they are all grouped
        together anyway.

        Peaks in `self.discarded_args` are still included here. This helps to filter away
        unwanted peaks later.
        """

        # `overlap_mask` is a square bool array.
        # `overlap_mask[i, j]` = do peaks[i] and peaks[j] overlap?
        n_peak = len(self._peaks_idx)
        self._overlap_mask = np.zeros((n_peak, n_peak))

        # use double loop to judge whether two peaks' region are overlapping
        for i in range(n_peak):
            self._overlap_mask[i][i] = 1
            mask1 = self._peak_masks[self._peaks_idx[i]]
            for j in range(i+1, n_peak):
                mask2 = self._peak_masks[self._peaks_idx[j]]
                if np.bitwise_and(mask1, mask2).any():
                    self._overlap_mask[i][j] = 1
                    self._overlap_mask[j][i] = 1

        # convert adjacency matrix to list of groups in which each element is overlapped
        flag = np.zeros(n_peak)
        for i in range(n_peak):
            if flag[i] == 1:
                continue
            indices = np.where(self._overlap_mask[i] == 1)
            group = np.unique(np.where(self._overlap_mask[indices] == 1)[1].reshape(-1))
            flag[indices] = 1
            self._peak_groups.append(group)