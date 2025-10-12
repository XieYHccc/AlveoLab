import time

import numpy as np
import trimesh as tm
from scipy.spatial import ConvexHull, QhullError

from AlveoLab.utils import get_logger, logging
from AlveoLab.orienter.pca_orienter import PcaOrienter
from AlveoLab.trimesh_utils import get_local_maximum_along_dir
from AlveoLab.segmentation.curvature_based_seg import CurvatureBasedSeg

logger = get_logger("landmark_recognizer.py", level=logging.DEBUG)

def _min_dist_points_to_segments(points, edges):
    """
    Params:
        P: (M,2) points
        edges: (K,2,2) edges

    return:
        (M,) min distance to hull for each points
    """
    A = edges[:, 0, :]   # (K,2)
    B = edges[:, 1, :]   # (K,2)
    AB = B - A           # (K,2)

    M = points.shape[0]
    K = A.shape[0]
    min_d2 = np.full(M, np.inf)

    for k in range(K):
        a = A[k]; ab = AB[k]
        ap = points - a                    # (M,2)
        denom = np.dot(ab, ab)
        if denom == 0.0:              # degenerate line segment
            d2 = np.sum((points - a)**2, axis=1)
        else:
            t = np.einsum('ij,j->i', ap, ab) / denom   # (M,)
            t = np.clip(t, 0.0, 1.0)
            proj = a + t[:, None] * ab                # (M,2)
            d2 = np.sum((points - proj)**2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return np.sqrt(min_d2)

class LandmarkRecognizer:
    _height_diff_threshold = 6.0  # mm, height difference from the highest point to height threshold
    _near_boundary_dist_threshold = 2.0  # mm, for filtering peaks near boundary

    def __init__(self, mesh, arch_type):
        self._mesh = mesh
        self._arch_type = arch_type
        self._orienter = None
        self._height_threshold = None
        self._peak_indices = None   # indices of peak vertices
        self._seg = None
        self._horizontal_hull = None
        self._near_boundary_peak_indices = None
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

    def _run(self):
        """
        Run all steps.
        """
        total_start = time.perf_counter()
        logger.info("[pipeline] LandmarkRecognizer run started")

        self._run_step(self._find_orientation,      "find_orientation")
        self._run_step(self._find_height_threshold, "find_height_threshold")
        self._run_step(self._find_peaks,            "find_peaks")
        self._run_step(self._segment_teeth,         "segment_teeth")

        total_elapsed = time.perf_counter() - total_start
        logger.info(f"[pipeline] all steps finished in {total_elapsed:.3f}s")

    def _find_orientation(self):
        self._orienter = PcaOrienter(self._mesh, self._arch_type)

    def _find_height_threshold(self):
        self._height_threshold = (np.inner(self._mesh.vertices,self._orienter.occlusal).max() -
                                  self._height_diff_threshold)

    def _find_peaks(self):
        # 1. get local maxima along occlusal direction
        peak_indices = get_local_maximum_along_dir(self._mesh, self._orienter.occlusal)
        # 2. filter peaks by height threshold
        peak_heights = np.inner(self._mesh.vertices[peak_indices], self._orienter.occlusal)
        peak_indices = peak_indices[peak_heights > self._height_threshold]
        # 3. remove peaks that on the boundary
        # ---------------------------------
        uv = np.c_[self._mesh.vertices @ self._orienter.right, self._mesh.vertices @ self._orienter.forward]
        # compute convex hull
        try:
            hull = ConvexHull(uv, qhull_options='QJ')
        except QhullError:
            self._peak_indices = peak_indices
            return

        loop = uv[hull.vertices]
        assert(len(loop) >= 3)
        edges = np.stack([loop, np.roll(loop, -1, axis=0)], axis=1)
        # compute min distance
        peaks = uv[peak_indices]
        d = _min_dist_points_to_segments(peaks, edges)  # (M,)
        keep_mask = d > 2.0

        # self._peak_indices = peak_indices
        self._peak_indices = peak_indices[keep_mask]
        self._horizontal_hull = hull.vertices
        self._near_boundary_peak_indices = peak_indices[~keep_mask]

    def _segment_teeth(self):
        self._seg = CurvatureBasedSeg(self._mesh, self._orienter, self._peak_indices)

    def _label_teeth(self):
        pass

    def _recognize_landmarks(self):
        pass


if __name__ == '__main__':
    mesh: tm.Trimesh = tm.load_mesh('../data/1JMandibular_export.stl')
    recognizer = LandmarkRecognizer(mesh, 'L')
