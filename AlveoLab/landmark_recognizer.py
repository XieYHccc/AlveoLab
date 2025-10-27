import time
import collections

import numpy as np
import trimesh as tm
from scipy.spatial import ConvexHull, QhullError

from AlveoLab.math.geometry import inner_product
from AlveoLab.utils import get_logger, logging, LazyAttribute
from AlveoLab.orienter.pca_dental_orienter import PcaOrienter
from AlveoLab.trimesh_utils import (get_local_maximum_along_dir, get_local_maximum,
                                    discrete_mean_curvature_measure, smooth_curvature)
from AlveoLab.segmentation.curvature_based_seg import CurvatureBasedSeg
from AlveoLab.segmentation.harmonic_based_seg import HarmonicBasedSeg

logger = get_logger("landmark_recognizer.py", level=logging.DEBUG)


def _min_dist_points_to_segments(points, edges):
    """
    Params:
        P: (M,2) points
        edges: (K,2,2) edges

    return:
        (M,) min distance to hull for each points
    """
    A = edges[:, 0, :]  # (K,2)
    B = edges[:, 1, :]  # (K,2)
    AB = B - A  # (K,2)

    M = points.shape[0]
    K = A.shape[0]
    min_d2 = np.full(M, np.inf)

    for k in range(K):
        a = A[k];
        ab = AB[k]
        ap = points - a  # (M,2)
        denom = np.dot(ab, ab)
        if denom == 0.0:  # degenerate line segment
            d2 = np.sum((points - a) ** 2, axis=1)
        else:
            t = np.einsum('ij,j->i', ap, ab) / denom  # (M,)
            t = np.clip(t, 0.0, 1.0)
            proj = a + t[:, None] * ab  # (M,2)
            d2 = np.sum((points - proj) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return np.sqrt(min_d2)


class LandmarkRecognizer:
    _HEIGHT_DIFF_THRESHOLD = 9.0  # mm, height difference from the highest point to height threshold
    _NEAR_BOUNDARY_DIST_THRESHOLD = 0.3  # mm, for filtering peaks near boundary
    _REMOVE_GUM_PEAKS_RATIO = 2  # vertical/horizontal ratio threshold for removing gum peaks

    def __init__(self, mesh, arch_type):
        self._mesh = mesh
        self._arch_type = arch_type
        self._orienter = None
        self.peak_indices = []  # indices of peak vertices
        self.discarded_peaks = collections.defaultdict(set)  # include "Too Low", "Near Boundary", "Gingiva Peaks"
        self.horizontal_hull = None
        self.height_threshold = 0.0

        self.seg = None

        self._run()

    @staticmethod
    def _run_step(func, name: str):
        start = time.perf_counter()
        logger.debug(f"[{name}] start")
        try:
            return func()
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.exception(f"[{name}] failed after {elapsed:.3f}s: {e}")
            raise
        finally:
            elapsed = time.perf_counter() - start
            logger.info(f"[{name}] done in {elapsed:.3f}s")


    @LazyAttribute
    def cutting_plane_intersect_vertices(self):
        cutting_plane_offset = self.height_threshold - 2 #TODO: adaptive offset
        vertex_heights = inner_product(self._mesh.vertices, self._orienter.occlusal)

        return  np.where((vertex_heights - cutting_plane_offset < 0.05) &
                         (vertex_heights - cutting_plane_offset > 0))[0]

    @LazyAttribute
    def vertex_mean_curvature(self):
        curvature = discrete_mean_curvature_measure(self._mesh)
        return curvature

    def _run(self):
        """
        Run all steps.
        """
        total_start = time.perf_counter()
        logger.info("[pipeline] LandmarkRecognizer run started")

        self._run_step(self._find_orientation, "find_orientation")

        self.height_threshold = (np.inner(self._mesh.vertices, self._orienter.occlusal).max() -
                            self._HEIGHT_DIFF_THRESHOLD)

        self._run_step(self._preprocess_mesh, "preprocess_mesh")
        self._run_step(self._find_peaks, "find_peaks")
        self._run_step(self._remove_peaks_near_boundary, "remove_peaks_near_boundary")
        self._run_step(self._remove_peaks_on_gingiva, "remove_peaks_on_gingiva")  #TODO: acutually not works well
        self._run_step(self._segment_teeth, "segment_teeth")

        total_elapsed = time.perf_counter() - total_start
        logger.info(f"[pipeline] all steps finished in {total_elapsed:.3f}s")

    def _find_orientation(self):
        self._orienter = PcaOrienter(self._mesh, self._arch_type)

    def _preprocess_mesh(self):
        cutting_plane_offset = self.height_threshold - 3 #TODO: adaptive offset
        faces_height = np.inner(self._mesh.triangles_center, self._orienter.occlusal)

        submeshes = self._mesh.submesh([faces_height > cutting_plane_offset], append=True).split(only_watertight=False)
        filtered = [m for m in submeshes if m.area > 20 and len(m.faces) > 100]
        assert len(filtered) > 0

        merged = tm.util.concatenate(filtered)
        assert len(merged.split(only_watertight=False)) == 1  # should be a single mesh now
        self._mesh = merged

    def _find_peaks(self):
        # 1. get local maxima along occlusal direction
        # heights = inner_product(self._mesh.vertices, self._orienter.occlusal)
        # curvature = smooth_curvature(self._mesh, self.vertex_mean_curvature, 3)
        # curvature = smooth_curvature(self._mesh, self.vertex_mean_curvature, 100)
        # K_norm = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature) + 1e-8)
        # z_norm = (heights - np.min(heights)) / (np.max(heights) - np.min(heights) + 1e-8)
        # H = 0.8 * z_norm + 0.2 * (-K_norm)
        # peak_indices = get_local_maximum(self._mesh, H)
        peak_indices = get_local_maximum_along_dir(self._mesh, self._orienter.occlusal)

        # 2. filter peaks by height threshold
        peak_heights = inner_product(self._mesh.vertices[peak_indices], self._orienter.occlusal)
        low_height_mask = peak_heights <= self.height_threshold
        self.discarded_peaks['Too Low'] = set(peak_indices[low_height_mask])
        self.peak_indices = peak_indices[~low_height_mask]

    def _remove_peaks_near_boundary(self):
        uv = np.c_[self._mesh.vertices @ self._orienter.right, self._mesh.vertices @ self._orienter.forward]
        # compute convex hull
        try:
            hull = ConvexHull(uv, qhull_options='QJ')
        except QhullError:
            return

        loop = uv[hull.vertices]
        assert (len(loop) >= 3)
        edges = np.stack([loop, np.roll(loop, -1, axis=0)], axis=1)
        # compute min distance
        peaks = uv[self.peak_indices]
        d = _min_dist_points_to_segments(peaks, edges)  # (M,)
        keep_mask = d > self._NEAR_BOUNDARY_DIST_THRESHOLD

        # self.peak_indices = peak_indices
        self.discarded_peaks['Near Boundary'] = set(self.peak_indices[~keep_mask])
        self.peak_indices = self.peak_indices[keep_mask]
        self._horizontal_hull = hull.vertices

    def _remove_peaks_on_gingiva(self):
        """
        Remove peaks that are likely gum peaks —
        if there exists another peak nearby (in horizontal plane)
        that is much higher vertically (vertical/horizontal ratio > 1)
        """
        peaks = np.array(self.peak_indices)
        vertices = self._mesh.vertices

        right_coords = vertices @ self._orienter.right
        forward_coords = vertices @ self._orienter.forward
        vertical_coords = vertices @ self._orienter.occlusal

        gingiva_peaks = set()
        for i, p1 in enumerate(peaks):
            if p1 in gingiva_peaks:
                continue
            for p2 in peaks[i + 1:]:
                dx = abs(right_coords[p1] - right_coords[p2])
                dy = abs(forward_coords[p1] - forward_coords[p2])
                if dx > 5.0 or dy > 5.0:
                    continue  # judge within a range
                if dy / dx > 2:
                    continue  # looks like peaks on different teeth(ugly!)

                # horizontal_dist = np.sqrt(dx * dx + dy * dy)
                # if horizontal_dist < 1e-6:
                #     continue
                vertical_dist = abs(vertical_coords[p1] - vertical_coords[p2])
                if vertical_dist < 2.0:
                    continue  # ignore small vertical difference
                if vertical_dist / dx > self._REMOVE_GUM_PEAKS_RATIO:
                    # lower peak is gum peak
                    lower_peak = p1 if vertical_coords[p1] < vertical_coords[p2] else p2
                    gingiva_peaks.add(lower_peak)

        # update
        self.discarded_peaks['Gingiva Peaks'] = gingiva_peaks
        self.peak_indices = np.array([p for p in peaks if p not in gingiva_peaks])

    def _segment_teeth(self):
        self.seg = CurvatureBasedSeg(self._mesh, self._orienter, self.peak_indices)
        self.discarded_peaks.update(self.seg.discarded_peaks)

        # update peak indices after segmentation, spilled peaks are removed
        self.peak_indices = self.seg.valid_peaks
        # self._group_region_mask = self.seg._group_region_masks

    def _label_teeth(self):
        pass

    def _recognize_landmarks(self):
        pass

    @LazyAttribute
    def harmonic_filed(self):
        discarded_peaks = []
        discarded_peaks.extend(set.union(*self.discarded_peaks.values()))
        discarded_peaks.extend(self.seg.discarded_peaks_all)
        discarded_peaks.extend(self.cutting_plane_intersect_vertices)

        self._harmonic_seg = HarmonicBasedSeg(
            self._mesh,
            teeth=self.seg.teeth,
            discarded_peaks=discarded_peaks,
            dental_quadratic = self.seg.quadratic
        )

        return self._harmonic_seg.harmonic_field


if __name__ == '__main__':
    mesh: tm.Trimesh = tm.load_mesh('../data/1JMandibular_export.stl')
    recognizer = LandmarkRecognizer(mesh, 'L')
