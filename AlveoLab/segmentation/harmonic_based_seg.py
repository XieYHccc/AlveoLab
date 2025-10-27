import scipy.sparse.linalg as spla

from AlveoLab.trimesh_utils import get_cotangent_weights_laplacian_matrix, build_Ab_from_L_and_constraints
from AlveoLab.utils import LazyAttribute


def solve_harmonic(A, b):
    # 最小二乘：解正规方程 (A^T A)Φ = A^T b，矩阵对称正定，适合 Cholesky
    ATA = A.T @ A
    ATb = A.T @ b
    # 用 SciPy 直接解；实际工程建议用 CHOLMOD（论文同源）以更快更稳
    return spla.spsolve(ATA.tocsc(), ATb)

class HarmonicBasedSeg:
    """
    To refine the tooth segmentation results from curvature-based segmentation
    """
    def __init__(self, dental_mesh, teeth, discarded_peaks, dental_quadratic):
        self.dental_mesh = dental_mesh
        self.teeth = teeth
        self.discarded_peaks = discarded_peaks
        self.dental_quadratic = dental_quadratic
        self.odd_teeth_args = []
        self.non_odd_teeth_args = []
        self.harmonic_field = None

        self._run()

    @LazyAttribute
    def laplacian_matrix(self):
        return get_cotangent_weights_laplacian_matrix(self.dental_mesh)

    def _run(self):
        self._sort_teeth()
        self._compute_harmonic_field()

    def _sort_teeth(self):
        """
        Sort teeth along the dental quadratic curve by the center of obb
        """

        tooth_roots = []
        for i, tooth in enumerate(self.teeth):
            center = tooth.obb.center
            root = self.dental_quadratic.get_root_at(center)
            tooth_roots.append((root, i))
        tooth_roots.sort(key=lambda x: x[0])
        self.odd_teeth_args  = [arg for idx, (_, arg) in enumerate(tooth_roots) if idx % 2 == 1]
        self.non_odd_teeth_args = [arg for idx, (_, arg) in enumerate(tooth_roots) if idx % 2 == 0]

    def _compute_harmonic_field(self):

        odd_teeth_peaks = []
        non_odd_teeth_peaks = []

        for i in self.odd_teeth_args:
            odd_teeth_peaks.extend(self.teeth[i].peaks)
        for i in self.non_odd_teeth_args:
            non_odd_teeth_peaks.extend(self.teeth[i].peaks)

        assert len(odd_teeth_peaks) > 1
        assert len(non_odd_teeth_peaks) > 1
        assert len(self.discarded_peaks) > 1

        A, b = build_Ab_from_L_and_constraints(
            self.laplacian_matrix,
            n=self.dental_mesh.vertices.shape[0],
            fs_idx=odd_teeth_peaks,
            bs_idx=non_odd_teeth_peaks,
            us_idx=self.discarded_peaks,
            w=1000.0
        )

        self.harmonic_field = solve_harmonic(A, b)




