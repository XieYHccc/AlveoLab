import time
import collections

import numpy as np
import trimesh as tm
from scipy.spatial import ConvexHull, QhullError

from AlveoLab.math.geometry import inner_product
from AlveoLab.utils import get_logger, logging, LazyAttribute
from AlveoLab.orienter.pca_dental_orienter import PcaOrienter
from AlveoLab.orienter.obb_dental_orienter import ObbOrienter
from AlveoLab.trimesh_utils import (get_local_maximum_along_dir, get_local_maximum,
                                    discrete_mean_curvature_measure, smooth_curvature)
from AlveoLab.segmentation.curvature_based_seg import CurvatureBasedSeg
from AlveoLab.segmentation.harmonic_based_seg import HarmonicBasedSeg
from AlveoLab.segmentation.label_arrays import build_face_and_vertex_label_arrays
from AlveoLab.mesh import Mesh
from AlveoLab.peak import Peak

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
    HEIGHT_DIFF_THRESHOLD = 7  # mm, height difference from the highest point to height threshold
    _NEAR_BOUNDARY_DIST_THRESHOLD = 0.3  # mm, for filtering peaks near boundary
    _REMOVE_GUM_PEAKS_RATIO = 2  # vertical/horizontal ratio threshold for removing gum peaks

    def __init__(self, mesh, arch_type):
        self.mesh = mesh
        self.arch_type = arch_type

        self.orienter = None
        self.horizontal_hull = None
        self.seg = None
        self.harmonic_seg = None
        self.teeth_face_labels = None
        self.teeth_vertex_labels = None

        self.height_threshold = 0.0
        self.peaks = []
        self.peak_indices = []  # indices of peak vertices
        self.discarded_peaks = collections.defaultdict(set)  # include "Too Low", "Near Boundary", "Gingiva Peaks"

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
        cutting_plane_offset = self.height_threshold- 2 #TODO: adaptive offset
        vertex_heights = inner_product(self.mesh.vertices, self.orienter.occlusal)

        return  np.where((vertex_heights - cutting_plane_offset < 0.05) &
                         (vertex_heights - cutting_plane_offset > 0))[0]

    @LazyAttribute
    def vertex_mean_curvature(self):
        curvature = discrete_mean_curvature_measure(self.mesh)
        return curvature

    @property
    def watershed_filtered_peaks(self):
        """All watershed-filtered peaks across every tooth, as a flat list."""
        return [p for tooth in self.teeth for p in tooth.filtered_peaks]
    
    def _run(self):
        """
        Run all steps.
        """
        total_start = time.perf_counter()
        logger.info("[pipeline] LandmarkRecognizer run started")

        self._run_step(self._find_orientation, "find_orientation")

        self.height_threshold = (np.inner(self.mesh.vertices, self.orienter.occlusal).max() -
                            self.HEIGHT_DIFF_THRESHOLD)
        # self._run_step(self._preprocess_mesh, "preprocess_mesh")
        self._run_step(self._find_peaks, "find_peaks")
        self._run_step(self._remove_peaks_near_boundary, "remove_peaks_near_boundary")
        self._run_step(self._remove_peaks_on_gingiva, "remove_peaks_on_gingiva")  #TODO: acutually not works well
        self._run_step(self._segment_teeth, "segment_teeth")
        self._run_step(self._filter_peaks_by_watershed, "filter_peaks_by_watershed")
        self._run_step(self._assemble_teeth_label_arrays, "assemble_teeth_labels")

        total_elapsed = time.perf_counter() - total_start
        logger.info(f"[pipeline] all steps finished in {total_elapsed:.3f}s")

    def _find_orientation(self):
        self.orienter = ObbOrienter(self.mesh, self.arch_type)

    def _preprocess_mesh(self):
        cutting_plane_offset = self.height_threshold - 4 #TODO: adaptive offset

        faces_height = np.inner(self.mesh.triangles_center, self.orienter.occlusal)

        submeshes = self.mesh.submesh([faces_height > cutting_plane_offset], append=True).split(only_watertight=False)
        filtered = [m for m in submeshes if m.area > 20 and len(m.faces) > 100]
        assert len(filtered) > 0

        merged = tm.util.concatenate(filtered)
        assert len(merged.split(only_watertight=False)) == 1  # should be a single mesh now
        self.mesh = Mesh(merged)

    def _find_peaks(self):
        # 1. get local maxima along occlusal direction
        # heights = inner_product(self.mesh.vertices, self.orienter.occlusal)
        # curvature = smooth_curvature(self.mesh, self.vertex_mean_curvature, 3)
        # curvature = smooth_curvature(self.mesh, self.vertex_mean_curvature, 100)
        # K_norm = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature) + 1e-8)
        # z_norm = (heights - np.min(heights)) / (np.max(heights) - np.min(heights) + 1e-8)
        # H = 0.8 * z_norm + 0.2 * (-K_norm)
        # peak_indices = get_local_maximum(self.mesh, H)
        peak_indices = get_local_maximum_along_dir(self.mesh, self.orienter.occlusal)

        # 2. filter peaks by height threshold
        peak_heights = inner_product(self.mesh.vertices[peak_indices], self.orienter.occlusal)
        low_height_mask = peak_heights <= self.height_threshold
        self.discarded_peaks['Too Low'] = set(peak_indices[low_height_mask])
        self.peak_indices = peak_indices[~low_height_mask]

        for idx in peak_indices[~low_height_mask]:
            self.peaks.append(Peak(self.mesh.vertices[idx], idx))
        self.peaks = np.array(self.peaks)

        self.discarded_peaks['Too Low'] = set()
        for idx in peak_indices[low_height_mask]:
            self.discarded_peaks['Too Low'].add(Peak(self.mesh.vertices[idx], idx))

    def _remove_peaks_near_boundary(self):
        uv = np.c_[self.mesh.vertices @ self.orienter.right, self.mesh.vertices @ self.orienter.forward]
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
        self.discarded_peaks['Near Boundary'] = set(self.peaks[~keep_mask])
        self.peaks = self.peaks[keep_mask]
        self.peak_indices = self.peak_indices[keep_mask]
        self.horizontal_hull = hull.vertices

    def _remove_peaks_on_gingiva(self):
        """
        Remove peaks that are likely gum peaks —
        if there exists another peak nearby (in horizontal plane)
        that is much higher vertically (vertical/horizontal ratio > 1)
        """
        peaks = np.array(self.peak_indices)
        vertices = self.mesh.vertices

        right_coords = vertices @ self.orienter.right
        forward_coords = vertices @ self.orienter.forward
        vertical_coords = vertices @ self.orienter.occlusal

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
        self.discarded_peaks['Gingiva Peaks'] = set([peak for peak in self.peaks if peak.index in gingiva_peaks])
        self.peak_indices = np.array([p for p in peaks if p not in gingiva_peaks])
        self.peaks = np.array([p for p in self.peaks if p.index not in gingiva_peaks])

    def _segment_teeth(self):
        self.seg = CurvatureBasedSeg(self.mesh, self.orienter, self.peaks)
        for reason, peaks in self.seg.discarded_peaks.items():
            self.discarded_peaks[reason].update(peaks)

        # update peak indices after segmentation, spilled peaks are removed
        self.peaks = self.seg.valid_peaks
        self.teeth = self.seg.teeth

    def _filter_peaks_by_watershed(self):
        for tooth in self.teeth:
            tooth.filter_peaks_watershed(self.orienter.occlusal)

    def _assemble_teeth_label_arrays(self):
        tooth_masks = [tooth.mask for tooth in self.teeth]
        tooth_keys = [tooth.key for tooth in self.teeth]
        self.teeth_face_labels, self.teeth_vertex_labels = build_face_and_vertex_label_arrays(
            self.mesh,
            tooth_masks,
            tooth_keys,
        )

    def _label_teeth(self):
        pass

    def _recognize_landmarks(self):
        pass

    @LazyAttribute
    def harmonic_field(self):
        non_tooth_point_indexes = []
        # for peaks in self.discarded_peaks.values():
        #     non_tooth_point_indexes.extend([peak.index for peak in peaks])

        # non_tooth_point_indexes.extend(set.union(*self.discarded_peaks.values()))
        # discarded_peaks.extend(self.seg.discarded_peaks_all)
        non_tooth_point_indexes.extend(self.cutting_plane_intersect_vertices)

        self.harmonic_seg = HarmonicBasedSeg(
            self.mesh,
            teeth=self.seg.teeth,
            non_tooth_point_indexes=non_tooth_point_indexes,
            dental_quadratic = self.seg.quadratic,
            orienter=self.orienter,
        )

        return self.harmonic_seg.harmonic_field

if __name__ == '__main__':
    mesh: tm.Trimesh = tm.load_mesh('../data/1JMandibular_export.stl')
    recognizer = LandmarkRecognizer(mesh, 'L')
