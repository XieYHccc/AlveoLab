"""
class for classifying a neonatal dental mesh into
unilateral(left or right) / bilateral and incomplete / complete
"""

import queue

import numpy as np
import networkx as nx
import trimesh as tm

from AlveoLab.utils import get_logger, logging, now
from AlveoLab.orienter.obb_orienter import ObbOrienter
from AlveoLab.trimesh_utils import get_face_face_adjacency
from AlveoLab.math.geometry import normalize_vector

logger = get_logger("cleft_classifier.py", level=logging.DEBUG)


class CleftClassifier:
    _MAX_COST = 2

    def __init__(self, mesh: tm.Trimesh):
        self._mesh = mesh.copy()
        self._face_face_adjacency = get_face_face_adjacency(self._mesh)
        # _, self._tri2tri_edge_curvature = get_edge_based_curvature(self._mesh, True)
        # self._tri2tri_costs = -self._tri2tri_edge_curvature.clip(max=0)

        self._cleft_position_mask = (0, 0)
        self._cleft_completeness_mask = (0, 0)
        self.peak_vertex_id_left = None
        self.peak_vertex_id_right = None
        self.peak_vertex_id_forward = None  # may not exist
        self.mask_left_segment = None
        self.mask_right_segment = None
        self.mask_forward_segment = None  # may not exist
        self.gap_landmark_vertex_id_left = None
        self.gap_landmark_vertex_id_right = None
        self.gap_landmark_vertex_id_forward_left = None
        self.gap_landmark_vertex_id_forward_right = None
        self.shortest_path_left_right = None
        self.shortest_path_left_forward = None
        self.shortest_path_right_forward = None

        self._run()

    @property
    def peak_ids(self):
        return self.peak_vertex_id_left, self.peak_vertex_id_right, self.peak_vertex_id_forward

    @property
    def cleft_position_mask(self):
        return self._cleft_position_mask

    @property
    def cleft_completeness_mask(self):
        return self._cleft_completeness_mask

    def print(self):
        print(self.cleft_position_mask)

    def _run(self):
        logger.info("vertices: %d, faces: %d", self._mesh.vertices.shape[0], self._mesh.faces.shape[0])
        start = now()

        # 1. orientate the mesh and move mesh to the origin
        step_start = now()
        self._orienter = ObbOrienter(self._mesh)
        self._mesh.apply_transform(self._orienter.to_origin_transform_matrix)
        step_end = now()
        logger.debug("Orient dental model in %0.4fs", step_end - step_start)

        # 2. find peaks
        self._find_peaks()

        # 3. perform region grow for left and right peak
        step_start = now()
        # try a small threshold first to see if it's a bilateral
        _, l = self._region_grow_from_peak(self.peak_vertex_id_left, 2)
        _, r = self._region_grow_from_peak(self.peak_vertex_id_right, 2)
        f = None
        if self.peak_vertex_id_forward is not None:
            _, f = self._region_grow_from_peak(self.peak_vertex_id_forward, 2)

            sum_fl = np.sum(l & f)
            sum_fr = np.sum(r & f)
            if sum_fl == 0 and sum_fr == 0:
                self._cleft_position_mask = (1, 1)
            elif sum_fr > 0 and sum_fl == 0:
                self._cleft_position_mask = (1, 0)
            elif sum_fl > 0 and sum_fr == 0:
                self._cleft_position_mask = (0, 1)
            else:
                logger.error("Cannot classify the cleft position")
        else:
            # compute sum of areas
            sum_l = np.sum(self._mesh.area_faces[l])
            sum_r = np.sum(self._mesh.area_faces[r])
            if sum_l > sum_r:
                self._cleft_position_mask = (0, 1)
            else:
                self._cleft_position_mask = (1, 0)

        # capture the alveolar segments with maximum precision
        assert self._cleft_position_mask != (0, 0)
        threshold = 2.7
        stop = False
        if self._cleft_position_mask == (1, 1):
            while not stop:
                _, self.mask_left_segment = self._region_grow_from_peak(self.peak_vertex_id_left, threshold)
                _, self.mask_right_segment = self._region_grow_from_peak(self.peak_vertex_id_right, threshold)
                _, self.mask_forward_segment = self._region_grow_from_peak(self.peak_vertex_id_forward, threshold)
                if np.sum(self.mask_left_segment & self.mask_forward_segment) > 0 or \
                        np.sum(self.mask_right_segment & self.mask_forward_segment) > 0:
                    threshold -= 0.1
                else:
                    stop = True
        else:
            while not stop:
                _, self.mask_left_segment = self._region_grow_from_peak(self.peak_vertex_id_left, threshold)
                _, self.mask_right_segment = self._region_grow_from_peak(self.peak_vertex_id_right, threshold)
                if np.sum(self.mask_left_segment & self.mask_right_segment) > 0:
                    threshold -= 0.1
                else:
                    stop = True

        # combine the forward segment in unilateral case
        if f is not None:
            if self.cleft_position_mask == (1, 0):
                self.mask_right_segment = self.mask_right_segment | f
            elif self.cleft_position_mask == (0, 1):
                self.mask_left_segment = self.mask_left_segment | f

        step_end = now()
        logger.debug("multi-stage region growing in %0.4fs", step_end - step_start)

        # threshold = 4.5
        # stop = False
        # while not stop:
        #     _, self.mask_left_segment = self._region_grow_from_peak(self.peak_vertex_id_left, threshold)
        #     b, self.mask_right_segment = self._region_grow_from_peak(self.peak_vertex_id_right, threshold,
        #                                                              self.mask_left_segment)
        #     if b is True:
        #         stop = True
        #     else:
        #         threshold -= 0.1
        #
        # # 4. perform region grow for forward peak and classify as unilateral/bilateral
        # if self.peak_vertex_id_forward is not None:
        #     # try a small threshold first to see if it's a bilateral
        #     combined_mask = self.mask_right_segment | self.mask_left_segment
        #     b, self.mask_forward_segment = self._region_grow_from_peak(self.peak_vertex_id_forward, 0.7, combined_mask)
        #     if b is True:
        #         # cleft is bilateral, refine the forward mask
        #         self._cleft_position_mask = (1, 1)
        #         threshold = 4.5
        #         stop = False
        #         while not stop:
        #             b, self.mask_forward_segment = self._region_grow_from_peak(self.peak_vertex_id_forward, threshold,
        #                                                                        combined_mask)
        #             if b is True:
        #                 stop = True
        #             else:
        #                 threshold -= 0.1
        #
        #     else:
        #         # cleft is unilateral, find in which side
        #         b_left, mask_left = self._region_grow_from_peak(self.peak_vertex_id_forward, 1.5,
        #                                                         self.mask_left_segment)
        #         b_right, mask_right = self._region_grow_from_peak(self.peak_vertex_id_forward, 1.5,
        #                                                           self.mask_right_segment)
        #         if b_left is True and b_right is False:
        #             # left side
        #             self._cleft_position_mask = (1, 0)
        #             _, self.mask_forward_segment = self._region_grow_from_peak(self.peak_vertex_id_forward, 2.0,
        #                                                                        self.mask_left_segment)
        #
        #             # merge mask
        #             self.mask_right_segment = self.mask_forward_segment | self.mask_right_segment
        #         else:
        #             # right side
        #             self._cleft_position_mask = (0, 1)
        #             _, self.mask_forward_segment = self._region_grow_from_peak(self.peak_vertex_id_forward, 2.0,
        #                                                                        self.mask_right_segment)
        #             # merge mask
        #             self.mask_left_segment = self.mask_forward_segment | self.mask_left_segment
        # else:
        #     # cleft is unilateral, and potentially complete
        #     if self.mask_left_segment.sum() > self.mask_right_segment.sum():
        #         self._cleft_position_mask = (1, 0)
        #     else:
        #         self._cleft_position_mask = (0, 1)

        # 5.locate landmarks near gaps
        self._locate_near_gap_landmarks()

        # 6.find the shortest path between gap landmarks
        step_start = now()
        self._find_shortest_path_between_gap_landmarks()
        step_end = now()
        logger.debug("Find shortest path in %0.4fs", step_end - step_start)

        # 7. determine cleft completeness
        self._determine_cleft_completeness()
        logger.debug("model's cleft position: %s", self.cleft_position_mask)
        logger.debug("model's cleft completeness: %s", self.cleft_completeness_mask)
        logger.debug("Classify dental model in %0.4fs", now() - start)

    def _find_peaks(self):
        # partition the mesh into three regions
        region_left = (self._mesh.vertices[:, 0] < -10) & (self._mesh.vertices[:, 2] < 5) & (
                    self._mesh.vertices[:, 2] > -5)
        region_right = (self._mesh.vertices[:, 0] > 10) & (self._mesh.vertices[:, 2] < 5) & (
                    self._mesh.vertices[:, 2] > -5)
        region_forward = (self._mesh.vertices[:, 2] > 5) & (np.abs(self._mesh.vertices[:, 0]) < 10)

        # find peaks in each region
        self.peak_vertex_id_left = np.where(region_left)[0][np.argmax(self._mesh.vertices[region_left, 1])]
        self.peak_vertex_id_right = np.where(region_right)[0][np.argmax(self._mesh.vertices[region_right, 1])]
        self.peak_vertex_id_forward = np.where(region_forward)[0][np.argmax(self._mesh.vertices[region_forward, 1])]

        # check if forward peak is valid by its local maximum property
        vertex_neighbors = self._mesh.vertex_neighbors[self.peak_vertex_id_forward]
        vertex_neighbor_points = self._mesh.vertices[vertex_neighbors]
        if np.any(vertex_neighbor_points[:, 1] > self._mesh.vertices[self.peak_vertex_id_forward][1]):
            self.peak_vertex_id_forward = None

    def _region_grow_from_peak(self, peak, threshold, checking_mask=None):
        assert peak is not None

        mask = np.full(self._mesh.faces.shape[0], False, dtype=bool)

        # init queue: push faces that contains peak
        q = queue.Queue()
        for f in self._mesh.vertex_faces[peak]:
            if f == -1:
                break
            q.put(f)
            mask[f] = True

        # begin region grow
        is_visited = np.full(self._mesh.faces.shape[0], False, dtype=bool)
        peak_height = self._mesh.vertices[peak][1]
        min_z = np.min(self._mesh.vertices[:, 2])
        while not q.empty():
            f = q.get()
            if checking_mask is not None and checking_mask[f]:
                return False, None
            if is_visited[f]:
                continue

            is_visited[f] = True

            face_center = self._mesh.triangles_center[f]
            face_center_height = face_center[1]
            if peak_height > face_center_height > peak_height - threshold and face_center[2] > min_z + 5:
                mask[f] = True
                face_neighbors = self._face_face_adjacency[f]
                for f_neighbor in face_neighbors:
                    if not is_visited[f_neighbor] and f_neighbor != -1:
                        q.put(f_neighbor)

        return True, mask

    def _region_grow_from_peak_with_curvature(self, peak, threshold, checking_mask=None):
        assert peak is not None

        # init queue
        peak_triangles = np.where(np.any(self._mesh.faces == peak, axis=1))
        accumulative_cost = np.ones(self._mesh.faces.shape[0]) * self._MAX_COST
        accumulative_cost[peak_triangles] = 0.0

        faces_init = np.unique(self._face_face_adjacency[peak_triangles].reshape(-1))
        que = queue.PriorityQueue()
        [que.put((accumulative_cost[face], face)) for face in faces_init]
        is_shortest = np.zeros(self._mesh.faces.shape[0], dtype=bool)

        # begin region growing
        peak_height = self._mesh.vertices[peak][1]
        while not que.empty():
            face = que.get()[1]
            if checking_mask is not None and checking_mask[face]:
                return False, None
            if is_shortest[face] == 1:
                continue

            is_shortest[face] = 1

            # make sure the region won't spread too deep
            face_center_height = self._mesh.triangles_center[face][1]
            if face_center_height > peak_height or face_center_height < peak_height - threshold:
                continue

            face_adj = self._face_face_adjacency[face]
            edge_costs_adj = self._tri2tri_costs[face]
            for i, neighbor_face in enumerate(face_adj):
                tmp_cost = accumulative_cost[face] + edge_costs_adj[i]
                if tmp_cost < accumulative_cost[neighbor_face]:
                    accumulative_cost[neighbor_face] = tmp_cost
                    que.put((accumulative_cost[neighbor_face], neighbor_face))

        return True, is_shortest

    def _locate_near_gap_landmarks(self):
        assert self.cleft_position_mask != (0, 0)

        if self.cleft_position_mask == (1, 1):  # bilateral
            # define two directions to find landmarks
            # dir_left_peak_to_forward_peak = self._mesh.vertices[self.peak_vertex_id_forward] - \
            #                                 self._mesh.vertices[self.peak_vertex_id_left]
            # dir_right_peak_to_forward_peak = self._mesh.vertices[self.peak_vertex_id_forward] - \
            #                                  self._mesh.vertices[self.peak_vertex_id_right]
            dir_left_peak_to_forward_peak = [1, 0, 1]
            dir_right_peak_to_forward_peak = [-1, 0, 1]
            dir_right_peak_to_forward_peak = normalize_vector(dir_right_peak_to_forward_peak)
            dir_left_peak_to_forward_peak = normalize_vector(dir_left_peak_to_forward_peak)

            # find landmarks
            left_segment_triangle_centers = self._mesh.triangles_center[self.mask_left_segment]
            right_segment_triangle_centers = self._mesh.triangles_center[self.mask_right_segment]
            forward_segment_triangle_centers = self._mesh.triangles_center[self.mask_forward_segment]

            left_side_dots = np.dot(left_segment_triangle_centers, dir_left_peak_to_forward_peak)
            right_side_dots = np.dot(right_segment_triangle_centers, dir_right_peak_to_forward_peak)
            forward_left_dots = np.dot(forward_segment_triangle_centers, -dir_left_peak_to_forward_peak)
            forward_right_dots = np.dot(forward_segment_triangle_centers, -dir_right_peak_to_forward_peak)
            landmark_face_id_left = np.where(self.mask_left_segment)[0][np.argmax(left_side_dots)]
            landmark_face_id_right = np.where(self.mask_right_segment)[0][np.argmax(right_side_dots)]
            landmark_face_id_forward_left = np.where(self.mask_forward_segment)[0][np.argmax(forward_left_dots)]
            landmark_face_id_forward_right = np.where(self.mask_forward_segment)[0][np.argmax(forward_right_dots)]
            self.gap_landmark_vertex_id_left = self._mesh.faces[landmark_face_id_left][0]
            self.gap_landmark_vertex_id_right = self._mesh.faces[landmark_face_id_right][0]
            self.gap_landmark_vertex_id_forward_left = self._mesh.faces[landmark_face_id_forward_left][0]
            self.gap_landmark_vertex_id_forward_right = self._mesh.faces[landmark_face_id_forward_right][0]
        elif self.cleft_position_mask == (1, 0):  # left side
            dir_1 = normalize_vector(np.array([1, 0, 1]))
            left_side_dots = np.dot(self._mesh.triangles_center[self.mask_left_segment], dir_1)
            landmark_face_id_left = np.where(self.mask_left_segment)[0][np.argmax(left_side_dots)]
            self.gap_landmark_vertex_id_left = self._mesh.faces[landmark_face_id_left][0]

            clipped_right_segment = self.mask_right_segment & (self._mesh.triangles_center[:, 2] > 5)
            right_side_dots = np.dot(self._mesh.triangles_center[clipped_right_segment], -dir_1)
            landmark_face_id_right = np.where(clipped_right_segment)[0][np.argmax(right_side_dots)]
            self.gap_landmark_vertex_id_right = self._mesh.faces[landmark_face_id_right][0]
        else:  # right side
            dir_1 = normalize_vector(np.array([-1, 0, 1]))
            right_side_dots = np.dot(self._mesh.triangles_center[self.mask_right_segment], dir_1)
            landmark_face_id_right = np.where(self.mask_right_segment)[0][np.argmax(right_side_dots)]
            self.gap_landmark_vertex_id_right = self._mesh.faces[landmark_face_id_right][0]

            clipped_left_segment = self.mask_left_segment & (self._mesh.triangles_center[:, 2] > 5)
            left_side_dots = np.dot(self._mesh.triangles_center[clipped_left_segment], -dir_1)
            landmark_face_id_left = np.where(clipped_left_segment)[0][np.argmax(left_side_dots)]
            self.gap_landmark_vertex_id_left = self._mesh.faces[landmark_face_id_left][0]

    def _find_shortest_path_between_gap_landmarks(self):
        assert self.gap_landmark_vertex_id_left is not None
        assert self.gap_landmark_vertex_id_right is not None

        # build a graph
        edges = self._mesh.edges_unique
        length = self._mesh.edges_unique_length
        g = nx.Graph()
        for e, l in zip(edges, length):
            g.add_edge(*e, weight=l)

        # find the shortest path
        if self._cleft_position_mask == (1, 1):  # bilateral
            self.shortest_path_left_forward = nx.shortest_path(g, source=self.gap_landmark_vertex_id_forward_left,
                                                               target=self.gap_landmark_vertex_id_left, weight='weight')
            self.shortest_path_right_forward = nx.shortest_path(g, source=self.gap_landmark_vertex_id_forward_right,
                                                                target=self.gap_landmark_vertex_id_right,
                                                                weight='weight')
        else:
            self.shortest_path_left_right = nx.shortest_path(g, source=self.gap_landmark_vertex_id_left,
                                                             target=self.gap_landmark_vertex_id_right, weight='weight')

    def _determine_cleft_completeness(self):
        if self.cleft_position_mask == (1, 1):  # bilateral
            left_completeness = False
            right_completeness = False

            # determine left side
            left_shortest_path = self._mesh.vertices[self.shortest_path_left_forward]
            min_height = np.min(left_shortest_path[:, 1])
            if self._mesh.vertices[self.gap_landmark_vertex_id_left][1] - min_height > 3 and \
                    self._mesh.vertices[self.gap_landmark_vertex_id_forward_left][1] - min_height > 3:
                left_completeness = True

            # determine right side
            right_shortest_path = self._mesh.vertices[self.shortest_path_right_forward]
            min_height = np.min(right_shortest_path[:, 1])
            if self._mesh.vertices[self.gap_landmark_vertex_id_right][1] - min_height > 3 and \
                    self._mesh.vertices[self.gap_landmark_vertex_id_forward_right][1] - min_height > 3:
                right_completeness = True

            if left_completeness and right_completeness:
                self._cleft_completeness_mask = (1, 1)
            elif left_completeness:
                self._cleft_completeness_mask = (1, 0)
            elif right_completeness:
                self._cleft_completeness_mask = (0, 1)
        else:  # unilateral
            complete = False
            shortest_path = self._mesh.vertices[self.shortest_path_left_right]
            min_height = np.min(shortest_path[:, 1])
            if self._mesh.vertices[self.gap_landmark_vertex_id_left][1] - min_height > 3 and \
                    self._mesh.vertices[self.gap_landmark_vertex_id_right][1] - min_height > 3:
                complete = True

            if complete:
                self._cleft_completeness_mask = (1, 0) if self.cleft_position_mask == (1, 0) else (0, 1)
