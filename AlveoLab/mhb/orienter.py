from __future__ import annotations

import numpy as np

from AlveoLab.math.geometry import normalize_vector
from AlveoLab.orienter._base_orienter import BaseOrienter
from AlveoLab.orienter._pca import Pca
from AlveoLab.orienter.pca_dental_orienter import PcaOrienter
from AlveoLab.utils import infer_arch_type


class ToothRegionPcaOrienter(BaseOrienter):
    """
    Estimate orientation from tooth-labeled regions instead of the full mesh.

    This helps reduce the influence of gingiva, cleft regions, and other
    non-tooth geometry when a per-vertex tooth segmentation is already
    available.
    """

    @property
    def right(self):
        return self._axisX

    @property
    def up(self):
        return self._axisY

    @property
    def forward(self):
        return self._axisZ

    @property
    def center(self):
        return self._center

    @property
    def occlusal(self):
        return self.up if self.arch_type == "L" else -self.up

    @property
    def axes(self) -> np.ndarray:
        return np.column_stack((self._axisX, self._axisY, self._axisZ))

    @property
    def to_origin_transform_matrix(self):
        result = np.eye(4)
        result[:3, :3] = self.axes
        result[:3, 3] = self.center
        if self.arch_type == "U":
            result[:3, :2] = -result[:3, :2]
        return np.linalg.inv(result)

    def __init__(
        self,
        mesh,
        vertex_labels,
        arch_type=None,
        *,
        tooth_labels: tuple[int, ...] | list[int] | np.ndarray | None = None,
        face_min_tooth_vertices: int = 2,
        min_selected_faces: int = 16,
        fallback_to_full_mesh: bool = True,
    ):
        super().__init__(mesh, arch_type)
        self.vertex_labels = np.asarray(vertex_labels, dtype=np.int64).reshape(-1)
        if self.vertex_labels.shape[0] != np.asarray(mesh.vertices).shape[0]:
            raise ValueError("vertex_labels length must match the mesh vertex count.")

        self.tooth_labels = None if tooth_labels is None else np.asarray(tooth_labels, dtype=np.int64).reshape(-1)
        self.face_min_tooth_vertices = int(face_min_tooth_vertices)
        self.min_selected_faces = int(min_selected_faces)
        self.fallback_to_full_mesh = fallback_to_full_mesh

        self._axisX = None
        self._axisY = None
        self._axisZ = None
        self._center = None

        self.tooth_vertex_mask = None
        self.selected_vertex_indices = None
        self.selected_face_mask = None
        self.selected_face_indices = None
        self.selected_face_centers = None
        self.selected_face_normals = None
        self.selected_face_areas = None
        self.fallback_orienter = None
        self.fallback_reason = None

        self._run()

    def _run(self):
        self._prepare_selection()

        if self._should_fallback():
            if not self.fallback_to_full_mesh:
                raise ValueError(self.fallback_reason or "Insufficient tooth-region geometry for orientation.")
            self._apply_fallback()
            return

        self._apply_pca()
        self._check_axis_y_sign()
        self._check_axis_z_sign()
        self._check_axis_x_sign()

        assert 1.001 > np.linalg.det(self.axes) > 0.999

    def _prepare_selection(self):
        if self.tooth_labels is None:
            self.tooth_vertex_mask = self.vertex_labels > 0
        else:
            self.tooth_vertex_mask = np.isin(self.vertex_labels, self.tooth_labels)

        self.selected_vertex_indices = np.flatnonzero(self.tooth_vertex_mask).astype(np.int64)

        face_tooth_counts = self.tooth_vertex_mask[np.asarray(self.mesh.faces)].sum(axis=1)
        self.selected_face_mask = face_tooth_counts >= self.face_min_tooth_vertices
        self.selected_face_indices = np.flatnonzero(self.selected_face_mask).astype(np.int64)
        self.selected_face_centers = np.asarray(self.mesh.triangles_center[self.selected_face_mask], dtype=float)
        self.selected_face_normals = np.asarray(self.mesh.face_normals[self.selected_face_mask], dtype=float)
        self.selected_face_areas = np.asarray(self.mesh.area_faces[self.selected_face_mask], dtype=float)

    def _should_fallback(self) -> bool:
        if self.selected_vertex_indices.size < 8:
            self.fallback_reason = "Too few tooth vertices for robust orientation."
            return True
        if self.selected_face_indices.size < self.min_selected_faces:
            self.fallback_reason = "Too few tooth faces for robust orientation."
            return True
        centered = self.selected_face_centers - self.selected_face_centers.mean(axis=0)
        if np.linalg.matrix_rank(centered) < 2:
            self.fallback_reason = "Tooth-region geometry is degenerate."
            return True
        return False

    def _apply_fallback(self):
        self.fallback_orienter = PcaOrienter(self.mesh, self.arch_type)
        self._axisX = np.asarray(self.fallback_orienter.right, dtype=float)
        self._axisY = np.asarray(self.fallback_orienter.up, dtype=float)
        self._axisZ = np.asarray(self.fallback_orienter.forward, dtype=float)
        self._center = np.asarray(self.fallback_orienter.center, dtype=float)

    def _small_face_weights(self):
        weights = (0.05 - self.selected_face_areas).clip(min=0.0)
        if not np.any(weights > 0):
            return None
        return weights

    def _apply_pca(self):
        pca = Pca(self.selected_face_centers, self._small_face_weights())
        self._center = np.asarray(pca.center_of_mass, dtype=float)
        self._axisY = normalize_vector(pca.eigenvectors[0])
        self._axisZ = normalize_vector(pca.eigenvectors[1])
        self._axisX = normalize_vector(pca.eigenvectors[2])

    def _approximate_occlusal_from_normals(self):
        if self.selected_face_normals.size:
            candidate = normalize_vector(self.selected_face_normals.sum(axis=0))
            if np.linalg.norm(candidate) > 1e-8:
                return candidate
        return normalize_vector(np.asarray(self.mesh.face_normals, dtype=float).sum(axis=0))

    def _check_axis_y_sign(self):
        approximated_occlusal = self._approximate_occlusal_from_normals()
        agreement = np.dot(approximated_occlusal, self.occlusal)
        if agreement == 0:
            agreement = 1.0
        self._axisY *= np.sign(agreement)

    def _check_axis_z_sign(self):
        x = np.dot(self.selected_face_centers - self.center, self._axisX)
        y = np.dot(self.selected_face_centers - self.center, self._axisZ)

        if x.size < 3:
            return

        weights = np.dot(self.selected_face_centers, self.occlusal)
        weights -= np.min(weights)
        if np.any(weights > 0):
            weights = weights ** 5
        else:
            weights = None

        try:
            poly = np.polynomial.Polynomial.fit(x, y, 2, w=weights)
        except Exception:
            return

        coef = poly.convert().coef
        if coef.shape[0] >= 3 and coef[2] > 0:
            self._axisZ = -self._axisZ

    def _check_axis_x_sign(self):
        self._axisX *= np.sign(np.linalg.det(self.axes))


if __name__ == "__main__":
    from pathlib import Path
    from types import SimpleNamespace

    import pyvista as pv

    from AlveoLab.mesh import Mesh
    from AlveoLab.mhb.medial_curve import build_arch_medial_curve
    from AlveoLab.pyvista_utils import add_direction_frame, get_dental_plotter, mesh_to_polydata
    from AlveoLab.utils import load_labels

    root = Path(__file__).resolve().parents[2]

    # Edit these values directly for quick manual debugging.
    mesh_path = root / "data" / "labeld_5year_betterv_objs" / "0580_5yr_Maxillary_export.obj"
    labels_path = root / "data" / "labeld_5year_betterv_objs" / "0580_5yr_Maxillary_export.json"
    arch_type = infer_arch_type(mesh_path)
    show_global_pca = True
    show_medial_curve = True

    mesh = Mesh.from_file(mesh_path)
    vertex_labels = load_labels(labels_path)
    orienter = ToothRegionPcaOrienter(mesh, vertex_labels, arch_type)

    frame = SimpleNamespace(
        right=orienter.right,
        forward=orienter.forward,
        occlusal=orienter.occlusal,
        center=orienter.center,
    )
    medial_curve = build_arch_medial_curve(mesh, vertex_labels, frame)

    print(f"Mesh: {mesh_path}")
    print(f"Labels: {labels_path}")
    print(f"Arch type: {arch_type}")
    print(f"Tooth vertices selected: {orienter.selected_vertex_indices.shape[0]}")
    print(f"Tooth faces selected: {orienter.selected_face_indices.shape[0]}")
    if orienter.fallback_orienter is not None:
        print(f"Fallback to full-mesh PCA: {orienter.fallback_reason}")
    else:
        print("Using tooth-region PCA orientation.")
    print(f"Right:    {orienter.right}")
    print(f"Forward:  {orienter.forward}")
    print(f"Occlusal: {orienter.occlusal}")

    plotter = get_dental_plotter()
    pv_mesh = mesh_to_polydata(mesh)

    base_color = np.array([0.86, 0.86, 0.86])
    tooth_color = np.array([0.96, 0.80, 0.42])
    colors = np.tile(base_color, (mesh.faces.shape[0], 1))
    colors[orienter.selected_face_mask] = tooth_color
    pv_mesh.cell_data["colors"] = colors

    plotter.add_mesh(
        pv_mesh,
        scalars="colors",
        rgb=True,
        opacity=1.0,
        specular=0.35,
        specular_power=8,
        ambient=0.25,
        show_edges=False,
    )

    axis_length = max(np.linalg.norm(np.ptp(np.asarray(mesh.vertices), axis=0)) * 0.12, 5.0)
    add_direction_frame(
        plotter,
        orienter.center,
        [
            ("right", orienter.right, "#d1495b"),
            ("forward", orienter.forward, "#00798c"),
            ("occlusal", orienter.occlusal, "#2a9d8f"),
        ],
        axis_length,
    )

    if show_global_pca:
        global_orienter = PcaOrienter(mesh, arch_type)
        add_direction_frame(
            plotter,
            global_orienter.center,
            [
                ("right", global_orienter.right, "#d1495b"),
                ("forward", global_orienter.forward, "#00798c"),
                ("occlusal", global_orienter.occlusal, "#2a9d8f"),
            ],
            axis_length * 0.8,
            opacity=0.35,
            prefix="global",
        )

    tooth_points = np.asarray(mesh.vertices[vertex_labels > 0], dtype=float)
    tooth_height = float(np.median(tooth_points @ orienter.occlusal)) if tooth_points.size else 0.0

    if show_medial_curve and medial_curve is not None:
        medial_points_3d = medial_curve.medial_points_3d + tooth_height * orienter.occlusal
        medial_polyline = pv.lines_from_points(medial_points_3d, close=False)
        plotter.add_mesh(medial_polyline, color="#43aa8b", line_width=4)
        plotter.add_points(medial_points_3d[::8], color="#43aa8b", point_size=8, render_points_as_spheres=True)

    tooth_face_centers = orienter.selected_face_centers
    if tooth_face_centers is not None and tooth_face_centers.size:
        plotter.add_points(
            tooth_face_centers[:: max(1, tooth_face_centers.shape[0] // 400)],
            color="#f4a261",
            point_size=5,
            render_points_as_spheres=True,
            opacity=0.65,
        )

    plotter.add_text(
        f"ToothRegionPcaOrienter ({arch_type})",
        position="upper_left",
        font_size=12,
        color="black",
    )
    plotter.show()
