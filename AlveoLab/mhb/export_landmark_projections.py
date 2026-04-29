from __future__ import annotations

import colorsys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AlveoLab.mesh import Mesh
from AlveoLab.mhb.labels import TOOTH_LABEL_DEFINITIONS
from AlveoLab.mhb.pipeline import MhbKeypointPipeline
from AlveoLab.utils import infer_arch_type, load_labels


def project_points_to_arch_plane(points: np.ndarray, frame) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return np.column_stack((points @ frame.right, points @ frame.forward))


def label_colors() -> dict[int, np.ndarray]:
    colors = {0: np.array([0.82, 0.82, 0.82], dtype=float)}
    for index, label in enumerate(sorted(TOOTH_LABEL_DEFINITIONS), start=0):
        colors[label] = np.array(colorsys.hsv_to_rgb(index / 12.0, 0.45, 0.88), dtype=float)
    return colors


def save_landmark_projection(
    mesh_path: Path,
    labels_path: Path,
    output_path: Path,
    *,
    map_labels: bool = True,
    point_stride: int = 8,
) -> None:
    mesh = Mesh.from_file(mesh_path)
    vertex_labels = load_labels(labels_path, map=map_labels)
    arch_type = infer_arch_type(mesh_path)
    pipeline = MhbKeypointPipeline(mesh, vertex_labels, arch_type=arch_type)

    colors = label_colors()
    tooth_mask = np.asarray(vertex_labels, dtype=np.int64) > 0
    tooth_points = np.asarray(mesh.vertices[tooth_mask], dtype=float)
    tooth_labels = np.asarray(vertex_labels[tooth_mask], dtype=np.int64)
    projection_2d = project_points_to_arch_plane(tooth_points, pipeline.global_frame)

    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    stride = max(int(point_stride), 1)
    for label in sorted(np.unique(tooth_labels)):
        label_mask = tooth_labels == int(label)
        label_points_2d = projection_2d[label_mask]
        if label_points_2d.size == 0:
            continue
        axis.scatter(
            label_points_2d[::stride, 0],
            label_points_2d[::stride, 1],
            s=5,
            c=[colors.get(int(label), colors[0])],
            alpha=0.45,
            linewidths=0,
        )

    cusp_points = []
    midpoint_points = []
    for result in pipeline.results.values():
        for keypoint in result.keypoints:
            projected = project_points_to_arch_plane(np.asarray(keypoint.point, dtype=float)[np.newaxis, :], pipeline.global_frame)[0]
            if keypoint.kind == "ridge_midpoint":
                midpoint_points.append(projected)
            elif keypoint.kind.startswith("cusp"):
                cusp_points.append(projected)

    has_cusps = bool(cusp_points)
    if has_cusps:
        cusp_points = np.asarray(cusp_points, dtype=float)
        axis.scatter(
            cusp_points[:, 0],
            cusp_points[:, 1],
            s=36,
            c="#2a9d8f",
            edgecolors="black",
            linewidths=0.35,
            label="cusps",
            zorder=4,
        )

    has_midpoints = bool(midpoint_points)
    if has_midpoints:
        midpoint_points = np.asarray(midpoint_points, dtype=float)
        axis.scatter(
            midpoint_points[:, 0],
            midpoint_points[:, 1],
            s=42,
            c="#d62828",
            edgecolors="black",
            linewidths=0.35,
            label="incisor midpoint",
            zorder=5,
        )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("right projection")
    axis.set_ylabel("forward projection")
    axis.set_title(mesh_path.stem)
    axis.grid(alpha=0.15)
    if has_cusps or has_midpoints:
        axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    # Edit these values directly for quick batch export.
    input_dir = root / "data" / "labeld_5year_betterv_objs"
    output_dir = root / "saved" / "new_cusps"
    map_labels = True
    point_stride = 8

    obj_paths = sorted(input_dir.glob("*.obj"))
    if not obj_paths:
        raise RuntimeError(f"No OBJ files were found in {input_dir}.")

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"OBJ count: {len(obj_paths)}")

    saved_count = 0
    skipped_count = 0
    failed_count = 0

    for mesh_path in obj_paths:
        labels_path = mesh_path.with_suffix(".json")
        if not labels_path.exists():
            print(f"[skip] Missing labels for {mesh_path.name}")
            skipped_count += 1
            continue

        output_path = output_dir / f"{mesh_path.stem}_landmarks.png"
        try:
            save_landmark_projection(
                mesh_path,
                labels_path,
                output_path,
                map_labels=map_labels,
                point_stride=point_stride,
            )
            print(f"[ok] {mesh_path.name} -> {output_path.name}")
            saved_count += 1
        except Exception as exc:
            print(f"[fail] {mesh_path.name}: {exc}")
            failed_count += 1

    print(f"Saved: {saved_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
