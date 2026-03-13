import trimesh as tm
import scipy.sparse.linalg as spla
import numpy as np

from AlveoLab.trimesh_utils import (get_cotangent_weights_laplacian_matrix, build_Ab_from_L_and_constraints,
                                    get_modified_cotangent_weights_laplacian_matrix)
from AlveoLab.utils import LazyAttribute
from AlveoLab.segmentation.isoline_voting import (
    extract_loop_candidates,
    compute_face_grad_magnitudes,
    pick_best_isoloops_per_tooth_dot_scissor,
    point_in_poly_2d,
)
from AlveoLab.mesh import Mesh
from AlveoLab.segmentation.tooth import Tooth
from AlveoLab.segmentation.cutting import find_optimal_gingiva_plane_trimesh_mean_paperlike
from AlveoLab.segmentation.extract_tooth_from_boundary import extract_tooth_submesh_from_candidate_faces

def crop_mesh_above_plane_and_remap(
    mesh: tm.Trimesh,
    occlusal: np.ndarray,
    z_cut: float,
    keep_above: bool = True,
    process: bool = False,
):
    """
    裁剪：保留三角面中心在 plane 上方的 faces，然后重建 mesh（顶点重编号）。
    返回：
      new_mesh: 裁剪后的 Trimesh
      old_to_new: (n_old,) int 映射数组，old_to_new[old_vid] = new_vid，若被裁掉则为 -1
      new_to_old: (n_new,) int 数组，new_to_old[new_vid] = old_vid
      kept_face_idx: 原 mesh 的 face indices（便于 debug / 进一步筛选组件）
    """
    occlusal = np.asarray(occlusal, dtype=float)
    occlusal /= (np.linalg.norm(occlusal) + 1e-12)

    # 以 face center 的高度判断保留哪些 faces
    face_h = mesh.triangles_center @ occlusal
    if keep_above:
        keep_face_mask = face_h > z_cut
    else:
        keep_face_mask = face_h < z_cut

    kept_face_idx = np.where(keep_face_mask)[0]
    if kept_face_idx.size == 0:
        raise RuntimeError("No faces kept after cropping. Check z_cut or direction.")

    # 取原 faces（旧顶点索引）
    F_keep_old = mesh.faces[kept_face_idx]  # (m,3) old vertex ids

    # 得到会用到的旧顶点集合
    used_old_vid = np.unique(F_keep_old.reshape(-1))
    used_old_vid = used_old_vid.astype(np.int64)

    # 构造 old->new 映射
    old_to_new = np.full(len(mesh.vertices), -1, dtype=np.int64)
    old_to_new[used_old_vid] = np.arange(len(used_old_vid), dtype=np.int64)

    # 新 faces
    F_new = old_to_new[F_keep_old]

    # 新 vertices
    V_new = mesh.vertices[used_old_vid]

    new_mesh = tm.Trimesh(vertices=V_new, faces=F_new, process=process)

    new_to_old = used_old_vid  # new_vid -> old_vid

    return new_mesh, old_to_new, new_to_old, kept_face_idx


def remap_vertex_indices(old_to_new: np.ndarray, idx: np.ndarray):
    """
    把旧顶点索引映射到新顶点索引，自动丢弃被裁掉的点（映射为 -1 的）。
    """
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    mapped = old_to_new[idx]
    mapped = mapped[mapped >= 0]
    return np.unique(mapped)


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

    def __init__(self, dental_mesh, teeth, non_tooth_point_indexes, dental_quadratic, orienter):
        self.dental_mesh = dental_mesh
        self.teeth = teeth
        self.non_tooth_point_indexes = non_tooth_point_indexes
        self.dental_quadratic = dental_quadratic
        self.orienter = orienter

        self.odd_teeth_peaks = []
        self.non_odd_teeth_peaks = []
        self.odd_teeth_args = []
        self.non_odd_teeth_args = []
        self.harmonic_field = None
        self.tooth_boundaries = []
        self.tooth_masks = []
        self.segmented_teeth = []

        self._run()

    @LazyAttribute
    def laplacian_matrix(self):
        return get_modified_cotangent_weights_laplacian_matrix(self.dental_mesh)

    @LazyAttribute
    def cutting_plane_intersect_vertices(self):
        pass

    def _run(self):
        self._sort_teeth()
        self._find_best_cutting_plane()
        self._compute_harmonic_field()
        self.tooth_boundaries = self.pick_best_isoloops_for_all_teeth(self.orienter.right, self.orienter.forward, n_isos=120, min_peak_ratio=0.6)
        # self.extract_all_teeth_meshes()

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
        self.odd_teeth_args = [arg for idx, (_, arg) in enumerate(tooth_roots) if idx % 2 == 1]
        self.non_odd_teeth_args = [arg for idx, (_, arg) in enumerate(tooth_roots) if idx % 2 == 0]

    def _compute_harmonic_field(self):

        odd_teeth_peaks = []
        even_teeth_peaks = []

        for i in self.odd_teeth_args:
            peak_indexes = [peak.index for peak in self.teeth[i].peaks]
            odd_teeth_peaks.extend(peak_indexes)
        for i in self.non_odd_teeth_args:
            peak_indexes = [peak.index for peak in self.teeth[i].peaks]
            even_teeth_peaks.extend(peak_indexes)

        assert len(odd_teeth_peaks) > 1
        assert len(even_teeth_peaks) > 1
        # assert len(self.non_tooth_point_indexes) > 1

        A, b = build_Ab_from_L_and_constraints(
            self.laplacian_matrix,
            n=self.dental_mesh.vertices.shape[0],
            fs_idx=odd_teeth_peaks,
            bs_idx=even_teeth_peaks,
            us_idx=self.gingiva_vertices_indexes,
            w=1000.0
        )

        self.harmonic_field = solve_harmonic(A, b)
        self.even_teeth_peaks = even_teeth_peaks
        self.odd_teeth_peaks = odd_teeth_peaks

    def _find_best_cutting_plane(self):
        z_final, gingiva_ring_vidx, dbg =  find_optimal_gingiva_plane_trimesh_mean_paperlike(
            mesh_wrap=self.dental_mesh,
            occlusal=self.orienter.occlusal,
            step_mm=0.5,
            top_margin_mm=3.0,
            close_tol= 1e-3
        )

        print("z_final:", z_final)
        print("candidate count", dbg["candidate_count"])

        self.debug = dbg
        self.cutting_plane_height = z_final
        self.gingiva_vertices_indexes = gingiva_ring_vidx

        new_mesh_tm, old_to_new, new_to_old, kept_faces = crop_mesh_above_plane_and_remap(
            mesh=self.dental_mesh,
            occlusal=self.orienter.occlusal,
            z_cut=self.cutting_plane_height,
            keep_above=True,
            process=False,
        )

        # 2) 把 trimesh -> 你的 Mesh 包装类（如果需要）
        self.dental_mesh = Mesh(new_mesh_tm)

        # 3) 映射 gingiva ring 顶点索引
        self.gingiva_vertices_indexes = remap_vertex_indices(old_to_new, self.gingiva_vertices_indexes)

        # 4) 映射 non_tooth_point_indexes（如果你还在用它）
        self.non_tooth_point_indexes = remap_vertex_indices(old_to_new, self.non_tooth_point_indexes)

        # 5) 最关键：映射所有 tooth peaks（因为 harmonic 约束 fs/bs 也要）
        for tooth in self.teeth:
            for peak in tooth.peaks:
                peak.index = int(old_to_new[peak.index])  # 若为 -1 说明 peak 被切掉，需要处理

    def pick_best_isoloops_for_all_teeth(
        self,
        right: np.ndarray,
        forward: np.ndarray,
        n_isos: int = 120,
        iso_percentile=(5.0, 95.0),
        stitch_tol: float = 1e-4,
        close_tol: float = 2e-3,
        min_points: int = 25,
        min_peak_ratio: float = 0.6,
        sigma_ms: float = 2.0,
        K_ms: int = 4,
        length_filter_ratio: float = 1.5,
    ):
        """
        Returns List[ToothBoundaryDotScissor] aligned with self.teeth order.
        Each entry has .best (LoopCandidate or None). LoopCandidate has poly3d + vote_score + iso.
        """
        V = np.asarray(self.dental_mesh.vertices, dtype=np.float64)
        F = np.asarray(self.dental_mesh.faces, dtype=np.int64)
        phi = np.asarray(self.harmonic_field, dtype=np.float64).reshape(-1)
        lower, upper = np.percentile(phi, [5, 95])
        # 将离群值clamp到这个范围
        hf_clamped = np.clip(phi, lower, upper)

        lo, hi = np.percentile(hf_clamped, list(iso_percentile))
        lo, hi = float(lo), float(hi)
        if hi - lo < 1e-12:
            raise RuntimeError("Harmonic field near-constant; cannot extract isoloops.")

        # 1) extract ALL loop candidates (closed components) across isovalues
        all_cands = extract_loop_candidates(
            V=V, F=F, phi=hf_clamped,
            right=right, forward=forward,
            lo=lo, hi=hi,
            n_isos=n_isos,
            stitch_tol=stitch_tol,
            close_tol=close_tol,
            min_points=min_points,
        )

        # 2) face weight g_i = normalized ||grad phi|| per face
        face_weight = compute_face_grad_magnitudes(V, F, phi)

        # 3) per-tooth peaks index list
        teeth_peaks_indices = []
        for tooth in self.teeth:
            teeth_peaks_indices.append([p.index for p in tooth.peaks])

        # 4) per-tooth Dot-Scissor selection
        results = pick_best_isoloops_per_tooth_dot_scissor(
            all_cands=all_cands,
            teeth_peaks_indices=teeth_peaks_indices,
            V=V,
            right=right,
            forward=forward,
            face_weight=face_weight,
            min_peak_ratio=min_peak_ratio,
            sigma_ms=sigma_ms,
            K_ms=K_ms,
            length_filter_ratio=length_filter_ratio,
        )
        return results

    def extract_all_teeth_meshes(self, store: bool = True):
        """
        按 self.teeth 的顺序提取每颗牙的网格，返回 List[Optional[tm.Trimesh]]。
        若某颗牙没有 boundary / peaks / 提取失败，则该位置为 None（但顺序不变）。
        """

        # 基本健壮性：tooth_boundaries 应该已在 _run() 中算好
        if not hasattr(self, "tooth_boundaries"):
            raise RuntimeError("self.tooth_boundaries not found. Make sure _run() has been executed.")

        tooth_meshes = []
        tooth_region_masks = []  # 可选：保留 region_mask（原 mesh 面的 bool mask），方便 debug

        for i, tooth in enumerate(self.teeth):
            tb = self.tooth_boundaries[i]
            cand = getattr(tb, "best", None)

            # 没有 loop candidate
            if cand is None or getattr(cand, "face_ids", None) is None or len(cand.face_ids) == 0:
                tooth_meshes.append(None)
                tooth_region_masks.append(None)
                continue

            # peaks 作为种子（你这套方法需要）
            peak_vidx = np.asarray([p.index for p in tooth.peaks], dtype=np.int64)
            if peak_vidx.size == 0:
                tooth_meshes.append(None)
                tooth_region_masks.append(None)
                continue

            tooth_tm, region_mask = extract_tooth_submesh_from_candidate_faces(
                mesh=self.dental_mesh,
                candidate=cand,
                tooth_peak_vidx=peak_vidx,
            )

            tooth_meshes.append(tooth_tm)
            tooth_region_masks.append(region_mask)

        if store:
            self.tooth_meshes = tooth_meshes
            self.tooth_region_masks = tooth_region_masks

        return tooth_meshes