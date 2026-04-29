from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from AlveoLab.math.geometry import normalize_vector
from AlveoLab.mhb.frame import GlobalFrame


def _project_points_to_frame_xy(points: np.ndarray, frame: GlobalFrame) -> np.ndarray:
    return np.column_stack((points @ frame.right, points @ frame.forward))


def _lift_frame_xy_points(
    points_2d: np.ndarray,
    frame: GlobalFrame,
    *,
    occlusal_offset: float = 0.0,
) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=float)
    if points_2d.size == 0:
        return np.empty((0, 3), dtype=float)

    return (
        points_2d[:, [0]] * frame.right[np.newaxis, :]
        + points_2d[:, [1]] * frame.forward[np.newaxis, :]
        + float(occlusal_offset) * frame.occlusal[np.newaxis, :]
    )


def _select_medial_candidates(
    points_2d: np.ndarray,
    point: np.ndarray,
    tangent: np.ndarray,
    *,
    strip_half_width: float,
    candidate_radius: float,
    min_points_per_strip: int,
) -> np.ndarray:
    offsets = points_2d - point[np.newaxis, :]
    tangent_offsets = offsets @ tangent
    radial_distances = np.linalg.norm(offsets, axis=1)

    nearby_mask = (np.abs(tangent_offsets) <= strip_half_width) & (radial_distances <= candidate_radius)
    nearby_points = points_2d[nearby_mask]

    if nearby_points.shape[0] < min_points_per_strip:
        order = np.lexsort((np.abs(tangent_offsets), radial_distances))
        take = min(min_points_per_strip, points_2d.shape[0])
        nearby_points = points_2d[order[:take]]

    return nearby_points


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0 or not np.any(mask):
        return []

    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=False)]


@dataclass
class ArchMedialCurve:
    frame: GlobalFrame
    fitted_curve_points_2d: np.ndarray
    medial_points_2d: np.ndarray
    strip_half_width: float
    candidate_radius: float
    sample_spacing: float

    @property
    def medial_points_3d(self) -> np.ndarray:
        return self.points_2d_to_3d(self.medial_points_2d)

    @property
    def fitted_curve_points_3d(self) -> np.ndarray:
        return self.points_2d_to_3d(self.fitted_curve_points_2d)

    def points_2d_to_3d(self, points_2d: np.ndarray, *, occlusal_offset: float = 0.0) -> np.ndarray:
        return _lift_frame_xy_points(points_2d, self.frame, occlusal_offset=occlusal_offset)

    def project_points_to_2d(self, points: np.ndarray) -> np.ndarray:
        return _project_points_to_frame_xy(np.asarray(points, dtype=float), self.frame)

    def _curve_segment_indices_for_tooth(
        self,
        tooth_points_2d: np.ndarray,
        *,
        min_segment_points: int = 5,
    ) -> np.ndarray:
        x_min, y_min = tooth_points_2d.min(axis=0)
        x_max, y_max = tooth_points_2d.max(axis=0)
        margin = max(2.0 * self.sample_spacing, 0.15 * max(x_max - x_min, y_max - y_min, 1.0))

        curve_points = self.medial_points_2d
        in_box = (
            (curve_points[:, 0] >= x_min - margin)
            & (curve_points[:, 0] <= x_max + margin)
            & (curve_points[:, 1] >= y_min - margin)
            & (curve_points[:, 1] <= y_max + margin)
        )

        tooth_center = tooth_points_2d.mean(axis=0)
        runs = _contiguous_true_runs(in_box)
        if runs:
            best_run: tuple[int, int] | None = None
            best_score: tuple[float, float, int] | None = None
            for start, end in runs:
                run_points = curve_points[start : end + 1]
                center_distance = float(np.min(np.linalg.norm(run_points - tooth_center[np.newaxis, :], axis=1)))
                run_length_score = -float(end - start + 1)
                score = (center_distance, run_length_score, start)
                if best_score is None or score < best_score:
                    best_score = score
                    best_run = (int(start), int(end))

            if best_run is not None:
                start, end = best_run
                run_indices = np.arange(start, end + 1, dtype=np.int64)
                if run_indices.size >= min_segment_points:
                    return run_indices

        closest_index = int(np.argmin(np.linalg.norm(curve_points - tooth_center[np.newaxis, :], axis=1)))
        tooth_span = max(
            float(np.ptp(tooth_points_2d[:, 0])),
            float(np.ptp(tooth_points_2d[:, 1])),
            self.sample_spacing,
        )
        half_window = max(int(np.ceil(0.5 * tooth_span / max(self.sample_spacing, 1e-6))), min_segment_points // 2)
        start = max(0, closest_index - half_window)
        end = min(curve_points.shape[0] - 1, closest_index + half_window)

        while end - start + 1 < min_segment_points and (start > 0 or end < curve_points.shape[0] - 1):
            if start > 0:
                start -= 1
            if end - start + 1 >= min_segment_points:
                break
            if end < curve_points.shape[0] - 1:
                end += 1

        return np.arange(start, end + 1, dtype=np.int64)

    def _curve_segment_for_tooth(self, tooth_points_2d: np.ndarray) -> np.ndarray:
        segment_indices = self._curve_segment_indices_for_tooth(tooth_points_2d)
        return self.medial_points_2d[segment_indices]

    def fit_tooth_mesiodistal_line(self, tooth_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tooth_points_2d = _project_points_to_frame_xy(tooth_points, self.frame)
        curve_segment = self._curve_segment_for_tooth(tooth_points_2d)
        if curve_segment.shape[0] < 2:
            raise ValueError("Need at least two medial-curve points to fit a tooth mesiodistal line.")

        line_center = curve_segment.mean(axis=0)
        centered = curve_segment - line_center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        mesiodistal_2d = normalize_vector(vh[0])

        chord = curve_segment[-1] - curve_segment[0]
        if np.linalg.norm(chord) > 1e-8 and float(np.dot(mesiodistal_2d, chord)) < 0.0:
            mesiodistal_2d = -mesiodistal_2d

        scalars = centered @ mesiodistal_2d
        line_segment = line_center + np.array([[scalars.min()], [scalars.max()]]) * mesiodistal_2d[np.newaxis, :]
        return line_segment, mesiodistal_2d

    def estimate_tooth_axes(self, tooth_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, mesiodistal_2d = self.fit_tooth_mesiodistal_line(tooth_points)

        mesiodistal = normalize_vector(
            mesiodistal_2d[0] * self.frame.right + mesiodistal_2d[1] * self.frame.forward
        )
        if np.dot(mesiodistal, self.frame.right) < 0:
            mesiodistal = -mesiodistal

        buccolingual = normalize_vector(np.cross(self.frame.occlusal, mesiodistal))
        if np.dot(buccolingual, self.frame.forward) < 0:
            buccolingual = -buccolingual
        mesiodistal = normalize_vector(np.cross(buccolingual, self.frame.occlusal))
        return mesiodistal, buccolingual


def build_arch_medial_curve(
    mesh,
    vertex_labels: np.ndarray,
    frame: GlobalFrame,
    *,
    sample_count: int = 256,
    strip_half_width: float | None = None,
    candidate_radius: float | None = None,
    min_points_per_strip: int = 8,
) -> ArchMedialCurve | None:
    tooth_mask = np.asarray(vertex_labels, dtype=np.int64).reshape(-1) > 0
    tooth_points = np.asarray(mesh.vertices[tooth_mask], dtype=float)
    if tooth_points.shape[0] < 16:
        return None

    tooth_points_2d = _project_points_to_frame_xy(tooth_points, frame)
    xs = tooth_points_2d[:, 0]
    ys = tooth_points_2d[:, 1]
    if np.ptp(xs) < 1e-6:
        return None

    poly = np.polynomial.Polynomial.fit(xs, ys, 3).convert()

    x_samples = np.linspace(xs.min(), xs.max(), int(sample_count))
    y_samples = poly(x_samples)
    fitted_curve = np.column_stack((x_samples, y_samples))

    sample_spacing = float(np.mean(np.diff(x_samples))) if x_samples.size > 1 else 1.0
    strip_half_width = strip_half_width or max(2.0 * sample_spacing, 0.5)
    arch_span_2d = float(np.linalg.norm(np.ptp(tooth_points_2d, axis=0)))
    candidate_radius = candidate_radius or max(
        8.0 * sample_spacing,
        4.0 * strip_half_width,
        0.15 * arch_span_2d,
        1.5,
    )

    medial_points = []
    deriv = poly.deriv()

    for x_value, y_value in fitted_curve:
        tangent = np.array([1.0, float(deriv(x_value))], dtype=float)
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        point = np.array([x_value, y_value], dtype=float)

        nearby_points = _select_medial_candidates(
            tooth_points_2d,
            point,
            tangent,
            strip_half_width=strip_half_width,
            candidate_radius=candidate_radius,
            min_points_per_strip=min_points_per_strip,
        )

        signed_along_normal = (nearby_points - point[np.newaxis, :]) @ normal
        medial_point = point + np.median(signed_along_normal) * normal
        medial_points.append(medial_point)

    return ArchMedialCurve(
        frame=frame,
        fitted_curve_points_2d=fitted_curve,
        medial_points_2d=np.asarray(medial_points, dtype=float),
        strip_half_width=float(strip_half_width),
        candidate_radius=float(candidate_radius),
        sample_spacing=sample_spacing,
    )


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    from AlveoLab.mesh import Mesh
    from AlveoLab.utils import infer_arch_type, load_labels

    root = Path(__file__).resolve().parents[2]

    # Edit these values directly for quick manual debugging.
    mesh_path = root / "data" / "labeld_5year_betterv_objs" / "0674_5 YR_Maxillary_export.obj"
    labels_path = root / "data" / "labeld_5year_betterv_objs" / "0674_5 YR_Maxillary_export.json"
    arch_type = infer_arch_type(mesh_path)
    show_3d = False

    mesh = Mesh.from_file(mesh_path)
    vertex_labels = load_labels(labels_path)
    frame = GlobalFrame.from_mesh(mesh, arch_type=arch_type, vertex_labels=vertex_labels)
    medial_curve = build_arch_medial_curve(mesh, vertex_labels, frame)

    if medial_curve is None:
        raise RuntimeError("Failed to build medial curve for the current mesh and labels.")

    tooth_points = np.asarray(mesh.vertices[np.asarray(vertex_labels) > 0], dtype=float)
    tooth_points_2d = _project_points_to_frame_xy(tooth_points, frame)
    tooth_labels = sorted(int(label) for label in np.unique(np.asarray(vertex_labels, dtype=np.int64)) if int(label) > 0)
    tooth_direction_segments: list[tuple[np.ndarray, np.ndarray, int]] = []
    for tooth_label in tooth_labels:
        label_mask = np.asarray(vertex_labels, dtype=np.int64) == tooth_label
        label_points = np.asarray(mesh.vertices[label_mask], dtype=float)
        if label_points.shape[0] < 3:
            continue

        try:
            _, buccolingual_axis = medial_curve.estimate_tooth_axes(label_points)
        except Exception:
            continue

        tooth_center = label_points.mean(axis=0)
        tooth_center_2d = _project_points_to_frame_xy(tooth_center[np.newaxis, :], frame)[0]
        buccolingual_2d = np.array(
            [
                float(np.dot(buccolingual_axis, frame.right)),
                float(np.dot(buccolingual_axis, frame.forward)),
            ],
            dtype=float,
        )
        norm = np.linalg.norm(buccolingual_2d)
        if norm < 1e-8:
            continue

        buccolingual_2d /= norm
        tooth_points_label_2d = _project_points_to_frame_xy(label_points, frame)
        tooth_span = max(float(np.ptp(tooth_points_label_2d[:, 0])), float(np.ptp(tooth_points_label_2d[:, 1])), 1.0)
        half_length = 0.35 * tooth_span
        segment_start = tooth_center_2d - half_length * buccolingual_2d
        segment_end = tooth_center_2d + half_length * buccolingual_2d
        tooth_direction_segments.append((segment_start, segment_end, tooth_label))

    print(f"Mesh: {mesh_path}")
    print(f"Labels: {labels_path}")
    print(f"Arch type: {arch_type}")
    print(f"Tooth points: {tooth_points.shape[0]}")
    print(f"Fitted curve points: {medial_curve.fitted_curve_points_2d.shape[0]}")
    print(f"Medial points: {medial_curve.medial_points_2d.shape[0]}")
    print(f"Strip half width: {medial_curve.strip_half_width:.4f}")
    print(f"Candidate radius: {medial_curve.candidate_radius:.4f}")
    print(f"Sample spacing: {medial_curve.sample_spacing:.4f}")

    figure_fit, axis_fit = plt.subplots(figsize=(9, 7))
    axis_fit.scatter(
        tooth_points_2d[:: max(1, tooth_points_2d.shape[0] // 4000), 0],
        tooth_points_2d[:: max(1, tooth_points_2d.shape[0] // 4000), 1],
        s=6,
        c="#9aa0a6",
        alpha=0.35,
    )
    axis_fit.plot(
        medial_curve.fitted_curve_points_2d[:, 0],
        medial_curve.fitted_curve_points_2d[:, 1],
        color="#f4a261",
        linewidth=2.5,
    )
    axis_fit.set_aspect("equal", adjustable="box")
    axis_fit.set_xlabel("right axis projection")
    axis_fit.set_ylabel("forward axis projection")
    axis_fit.grid(alpha=0.2)
    figure_fit.tight_layout()

    figure_medial, axis_medial = plt.subplots(figsize=(9, 7))
    axis_medial.scatter(
        tooth_points_2d[:: max(1, tooth_points_2d.shape[0] // 4000), 0],
        tooth_points_2d[:: max(1, tooth_points_2d.shape[0] // 4000), 1],
        s=6,
        c="#9aa0a6",
        alpha=0.35,
    )
    axis_medial.plot(
        medial_curve.medial_points_2d[:, 0],
        medial_curve.medial_points_2d[:, 1],
        color="#2a9d8f",
        linewidth=3.2,
    )
    for segment_start, segment_end, _ in tooth_direction_segments:
        axis_medial.plot(
            [segment_start[0], segment_end[0]],
            [segment_start[1], segment_end[1]],
            color="#264653",
            linewidth=2.0,
            alpha=0.9,
        )
    axis_medial.set_aspect("equal", adjustable="box")
    axis_medial.set_xlabel("right axis projection")
    axis_medial.set_ylabel("forward axis projection")
    axis_medial.grid(alpha=0.2)
    figure_medial.tight_layout()
    plt.show()

    if show_3d:
        import pyvista as pv

        from AlveoLab.pyvista_utils import add_direction_frame, get_dental_plotter, mesh_to_polydata

        tooth_height = float(np.median(tooth_points @ frame.occlusal)) if tooth_points.size else 0.0
        medial_points_3d = medial_curve.points_2d_to_3d(
            medial_curve.medial_points_2d,
            occlusal_offset=tooth_height,
        )

        plotter = get_dental_plotter()
        pv_mesh = mesh_to_polydata(mesh)
        base_color = np.array([0.86, 0.86, 0.86])
        tooth_color = np.array([0.96, 0.80, 0.42])
        tooth_mask = np.asarray(vertex_labels) > 0
        face_tooth_counts = tooth_mask[np.asarray(mesh.faces)].sum(axis=1)
        colors = np.tile(base_color, (mesh.faces.shape[0], 1))
        colors[face_tooth_counts >= 2] = tooth_color
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
        plotter.add_mesh(pv.lines_from_points(medial_points_3d, close=False), color="#2a9d8f", line_width=5)
        plotter.add_points(
            medial_points_3d[:: max(1, medial_points_3d.shape[0] // 128)],
            color="#3d5a80",
            point_size=8,
            render_points_as_spheres=True,
        )
        axis_length = max(np.linalg.norm(np.ptp(np.asarray(mesh.vertices), axis=0)) * 0.12, 5.0)
        add_direction_frame(
            plotter,
            frame.center,
            [
                ("right", frame.right, "#d1495b"),
                ("forward", frame.forward, "#00798c"),
                ("occlusal", frame.occlusal, "#2a9d8f"),
            ],
            axis_length,
        )
        plotter.add_text("3D Medial Curve", position="upper_left", font_size=12, color="black")
        plotter.show()
