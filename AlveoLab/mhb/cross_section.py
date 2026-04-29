from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh as tm

from AlveoLab.mhb.models import ToothContext


@dataclass
class ToothCrossSection:
    orientation: str
    section_index: int
    offset: float
    plane_origin: np.ndarray
    plane_normal: np.ndarray
    in_plane_axis: np.ndarray
    curve_3d: np.ndarray
    curve_2d: np.ndarray
    status: str


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0 or not np.any(mask):
        return []

    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=False)]


def polyline_signed_curvature(points_2d: np.ndarray, smoothing_window: int = 5) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=float)
    if points_2d.ndim != 2 or points_2d.shape[0] < 3:
        return np.zeros(points_2d.shape[0], dtype=float)

    if smoothing_window >= 3 and points_2d.shape[0] >= smoothing_window:
        pad = smoothing_window // 2
        kernel = np.ones(smoothing_window, dtype=float) / float(smoothing_window)
        smoothed = np.empty_like(points_2d)
        for axis in range(2):
            padded = np.pad(points_2d[:, axis], (pad, pad), mode="edge")
            smoothed[:, axis] = np.convolve(padded, kernel, mode="valid")
        points_2d = smoothed

    prev_vectors = points_2d[1:-1] - points_2d[:-2]
    next_vectors = points_2d[2:] - points_2d[1:-1]
    prev_lengths = np.linalg.norm(prev_vectors, axis=1)
    next_lengths = np.linalg.norm(next_vectors, axis=1)
    length_scale = np.maximum(0.5 * (prev_lengths + next_lengths), 1e-8)

    cross_values = prev_vectors[:, 0] * next_vectors[:, 1] - prev_vectors[:, 1] * next_vectors[:, 0]
    dot_values = np.einsum("ij,ij->i", prev_vectors, next_vectors)
    turning_angles = np.arctan2(cross_values, dot_values)

    curvature = np.zeros(points_2d.shape[0], dtype=float)
    curvature[1:-1] = turning_angles / length_scale
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


def normalized_polyline_curvature(points_2d: np.ndarray, smoothing_window: int = 5) -> np.ndarray:
    curvature = polyline_signed_curvature(points_2d, smoothing_window=smoothing_window)
    max_abs = float(np.max(np.abs(curvature))) if curvature.size else 0.0
    if max_abs < 1e-8:
        return np.zeros_like(curvature)
    return curvature / max_abs


def _build_tooth_trimesh(context: ToothContext) -> tm.Trimesh:
    faces = np.asarray(context.mesh.faces[context.face_indices], dtype=np.int64)
    if faces.size == 0:
        return tm.Trimesh(vertices=np.asarray(context.points, dtype=float), faces=np.empty((0, 3), dtype=np.int64), process=False)

    unique_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    sub_vertices = np.asarray(context.mesh.vertices[unique_vertices], dtype=float)
    sub_faces = inverse.reshape(-1, 3)
    return tm.Trimesh(vertices=sub_vertices, faces=sub_faces, process=False)


def _sample_section_offsets(values: np.ndarray, count: int, margin_ratio: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=float)

    value_min = float(values.min())
    value_max = float(values.max())
    if value_max - value_min < 1e-8:
        return np.array([float(values.mean())], dtype=float)

    margin = float(margin_ratio) * (value_max - value_min)
    start = value_min + margin
    end = value_max - margin
    if end <= start:
        start = value_min
        end = value_max

    return np.linspace(start, end, int(max(count, 2)), dtype=float)


def _select_longest_section_curve(section, min_curve_points: int) -> tuple[np.ndarray, str]:
    curves = getattr(section, "discrete", None)
    if not curves:
        vertices = np.asarray(getattr(section, "vertices", np.empty((0, 3))), dtype=float)
        entities = list(getattr(section, "entities", []))
        scale = float(getattr(section, "scale", 1.0))
        fallback_curves = []
        for entity in entities:
            try:
                curve = np.asarray(entity.discrete(vertices, scale=scale), dtype=float)
            except TypeError:
                try:
                    curve = np.asarray(entity.discrete(vertices), dtype=float)
                except Exception:
                    continue
            except Exception:
                continue

            if curve.ndim == 2 and curve.shape[1] == 3 and curve.shape[0] >= 2:
                fallback_curves.append(curve)

        curves = fallback_curves
        if not curves:
            return np.empty((0, 3), dtype=float), "no_discrete_curve"

    best_curve = np.empty((0, 3), dtype=float)
    best_length = -1.0
    for curve in curves:
        curve = np.asarray(curve, dtype=float)
        if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 2:
            continue
        curve_length = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
        if curve_length > best_length:
            best_length = curve_length
            best_curve = curve

    if best_curve.shape[0] == 0:
        return np.empty((0, 3), dtype=float), "no_valid_curve"
    if best_curve.shape[0] < int(min_curve_points):
        return best_curve, "curve_too_short"
    return best_curve, "ok"


def build_tooth_cross_sections(
    context: ToothContext,
    *,
    orientation: str,
    count: int = 11,
    margin_ratio: float = 0.08,
    min_curve_points: int = 5,
) -> list[ToothCrossSection]:
    if orientation not in {"buccolingual", "mesiodistal"}:
        raise ValueError(f"Unsupported cross-section orientation: {orientation!r}")

    tooth_mesh = _build_tooth_trimesh(context)
    if orientation == "buccolingual":
        offsets = _sample_section_offsets(context.mesiodistal_values, count, margin_ratio)
        plane_normal = np.asarray(context.mesiodistal_axis, dtype=float)
        in_plane_axis = np.asarray(context.buccolingual_axis, dtype=float)
    else:
        offsets = _sample_section_offsets(context.buccolingual_values, count, margin_ratio)
        plane_normal = np.asarray(context.buccolingual_axis, dtype=float)
        in_plane_axis = np.asarray(context.mesiodistal_axis, dtype=float)

    if offsets.size == 0:
        return []

    center_projection = float(np.dot(context.center, plane_normal))
    sections: list[ToothCrossSection] = []
    for section_index, offset in enumerate(offsets):
        plane_origin = np.asarray(context.center, dtype=float) + (float(offset) - center_projection) * plane_normal
        section = tooth_mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if section is None:
            sections.append(
                ToothCrossSection(
                    orientation=orientation,
                    section_index=section_index,
                    offset=float(offset),
                    plane_origin=plane_origin,
                    plane_normal=plane_normal,
                    in_plane_axis=in_plane_axis,
                    curve_3d=np.empty((0, 3), dtype=float),
                    curve_2d=np.empty((0, 2), dtype=float),
                    status="no_intersection",
                )
            )
            continue

        curve_3d, status = _select_longest_section_curve(section, min_curve_points=min_curve_points)
        if status == "ok":
            curve_2d = np.column_stack((curve_3d @ in_plane_axis, curve_3d @ context.occlusal_axis))
        else:
            curve_2d = np.empty((0, 2), dtype=float)

        sections.append(
            ToothCrossSection(
                orientation=orientation,
                section_index=section_index,
                offset=float(offset),
                plane_origin=plane_origin,
                plane_normal=plane_normal,
                in_plane_axis=in_plane_axis,
                curve_3d=curve_3d,
                curve_2d=curve_2d,
                status=status,
            )
        )

    return sections


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    from AlveoLab.mesh import Mesh
    from AlveoLab.mhb.pipeline import MhbKeypointPipeline
    from AlveoLab.utils import infer_arch_type, load_labels

    root = Path(__file__).resolve().parents[2]

    # Edit these values directly for quick manual debugging.
    mesh_path = root / "data" / "labeld_5year_betterv_objs" / "VAL5_UpperJaw_030919.obj"
    labels_path = root / "saved" / "pred_labels_tgroupnet0302" / "VAL5_UpperJaw_030919.json"
    arch_type = infer_arch_type(mesh_path)
    tooth_label = 1
    orientation = "buccolingual"
    section_count = 11
    section_margin_ratio = 0.08
    map_labels = False
    show_3d = False

    mesh = Mesh.from_file(mesh_path)
    vertex_labels = load_labels(labels_path, map=map_labels)
    pipeline = MhbKeypointPipeline(mesh, vertex_labels, arch_type=arch_type)

    if tooth_label not in pipeline.contexts:
        raise ValueError(f"Tooth label {tooth_label} was not found in the current mesh/labels.")

    context = pipeline.contexts[tooth_label]
    sections = build_tooth_cross_sections(
        context,
        orientation=orientation,
        count=section_count,
        margin_ratio=section_margin_ratio,
        min_curve_points=5,
    )
    if not sections:
        raise RuntimeError("No cross-sections were generated for the current tooth.")

    middle_section_index = 5
    usable_sections = [section for section in sections if section.status == "ok" and np.asarray(section.curve_2d).shape[0] >= 2]
    if usable_sections:
        middle_section = min(usable_sections, key=lambda section: abs(int(section.section_index) - middle_section_index))
    else:
        middle_section = sections[middle_section_index]

    print(f"Mesh: {mesh_path}")
    print(f"Labels: {labels_path}")
    print(f"Arch type: {arch_type}")
    print(f"Tooth label: {tooth_label} ({context.definition.name})")
    print(f"Orientation: {orientation}")
    print(f"Axis source: {context.axis_source}")
    print(f"Section count: {len(sections)}")
    print("Section statuses:", ", ".join(f"{section.section_index}:{section.status}" for section in sections))
    print(f"Requested middle index: {middle_section_index}")
    print(f"Displayed section index: {middle_section.section_index}")
    print(f"Middle section status: {middle_section.status}")
    print(f"Middle section offset: {middle_section.offset:.4f}")
    print(f"Middle section plane origin: {middle_section.plane_origin}")
    print(f"Middle section plane normal: {middle_section.plane_normal}")

    if middle_section.status != "ok":
        raise RuntimeError(
            f"No usable section was found near the middle: displayed status={middle_section.status!r}. "
            "You can change tooth_label / orientation / section_count and try again."
        )

    curve_2d = np.asarray(middle_section.curve_2d, dtype=float)
    curvature = normalized_polyline_curvature(curve_2d, smoothing_window=5)

    figure, axis = plt.subplots(figsize=(7.0, 5.5))
    axis.plot(curve_2d[:, 0], curve_2d[:, 1], color="#7a7a7a", linewidth=1.8, alpha=0.9)
    scatter = axis.scatter(
        curve_2d[:, 0],
        curve_2d[:, 1],
        c=curvature,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        s=28,
    )
    axis.set_title(
        f"{context.definition.name} | {orientation} middle section #{middle_section_index}",
        fontsize=12,
    )
    axis.set_xlabel("in-plane axis")
    axis.set_ylabel("occlusal")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("normalized 2D curvature")
    figure.tight_layout()
    plt.show()

    if show_3d:
        import pyvista as pv

        from AlveoLab.pyvista_utils import get_dental_plotter, mesh_to_polydata

        plotter = get_dental_plotter()
        pv_mesh = mesh_to_polydata(mesh)
        plotter.add_mesh(
            pv_mesh,
            color="#d9d9d9",
            opacity=0.45,
            specular=0.25,
            ambient=0.3,
            show_edges=False,
        )

        tooth_faces = np.asarray(mesh.faces[context.face_indices], dtype=np.int64)
        if tooth_faces.size:
            tooth_mesh = Mesh.from_vertices_faces(np.asarray(mesh.vertices, dtype=float), tooth_faces)
            plotter.add_mesh(mesh_to_polydata(tooth_mesh), color="#f4a261", opacity=0.85, show_edges=False)

        plane_extent = max(
            float(np.ptp(context.buccolingual_values)),
            float(np.ptp(context.mesiodistal_values)),
            float(np.ptp(context.occlusal_values)),
            1.0,
        )
        plane = pv.Plane(
            center=np.asarray(middle_section.plane_origin, dtype=float),
            direction=np.asarray(middle_section.plane_normal, dtype=float),
            i_size=1.8 * plane_extent,
            j_size=1.8 * plane_extent,
        )
        plotter.add_mesh(plane, color="#4cc9f0", opacity=0.18)
        plotter.add_mesh(pv.lines_from_points(np.asarray(middle_section.curve_3d, dtype=float), close=False), color="#d62828", line_width=6)
        plotter.add_points(
            np.asarray(middle_section.curve_3d, dtype=float),
            color="#d62828",
            point_size=8,
            render_points_as_spheres=True,
        )
        plotter.add_text("Middle Cross-Section", position="upper_left", font_size=12, color="black")
        plotter.show()
