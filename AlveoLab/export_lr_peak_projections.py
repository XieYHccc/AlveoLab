from __future__ import annotations

import collections
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AlveoLab.landmark_recognizer import LandmarkRecognizer
from AlveoLab.mesh import Mesh
from AlveoLab.utils import infer_arch_type, load_labels


def project_points_to_arch_plane(points: np.ndarray, recognizer: LandmarkRecognizer) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return np.column_stack(
        (
            points @ np.asarray(recognizer.orienter.right, dtype=float),
            points @ np.asarray(recognizer.orienter.forward, dtype=float),
        )
    )


def build_partial_recognizer(mesh: Mesh, arch_type: str) -> LandmarkRecognizer:
    recognizer = LandmarkRecognizer.__new__(LandmarkRecognizer)
    recognizer.mesh = mesh
    recognizer.arch_type = arch_type

    recognizer.orienter = None
    recognizer.horizontal_hull = None
    recognizer.seg = None
    recognizer.harmonic_seg = None
    recognizer.teeth_face_labels = None
    recognizer.teeth_vertex_labels = None

    recognizer.height_threshold = 0.0
    recognizer.peaks = []
    recognizer.peak_indices = []
    recognizer.discarded_peaks = collections.defaultdict(set)

    recognizer._find_orientation()
    recognizer.height_threshold = (
        np.inner(recognizer.mesh.vertices, recognizer.orienter.occlusal).max()
        - recognizer.HEIGHT_DIFF_THRESHOLD
    )
    recognizer._find_peaks()
    recognizer._remove_peaks_near_boundary()
    recognizer._remove_peaks_on_gingiva()
    return recognizer


def save_lr_peak_projection(
    mesh_path: Path,
    output_path: Path,
    *,
    point_stride: int = 8,
) -> None:
    mesh = Mesh.from_file(mesh_path)
    arch_type = infer_arch_type(mesh_path)
    labels_path = mesh_path.with_suffix(".json")
    fallback_message = None

    try:
        recognizer = LandmarkRecognizer(mesh, arch_type)
        tooth_mask = np.asarray(recognizer.teeth_vertex_labels, dtype=np.int64) > 0
    except Exception as recognizer_error:
        recognizer = build_partial_recognizer(mesh, arch_type)
        fallback_message = str(recognizer_error)
        if labels_path.exists():
            tooth_mask = np.asarray(load_labels(labels_path, map=True), dtype=np.int64) > 0
        else:
            tooth_mask = np.ones(len(mesh.vertices), dtype=bool)

    if np.any(tooth_mask):
        tooth_points = np.asarray(mesh.vertices[tooth_mask], dtype=float)
    else:
        tooth_points = np.asarray(mesh.vertices, dtype=float)

    tooth_projection = project_points_to_arch_plane(tooth_points, recognizer)
    peak_points = np.asarray([peak.point for peak in recognizer.peaks], dtype=float)
    peak_projection = (
        project_points_to_arch_plane(peak_points, recognizer)
        if peak_points.size
        else np.empty((0, 2), dtype=float)
    )

    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    stride = max(int(point_stride), 1)
    axis.scatter(
        tooth_projection[::stride, 0],
        tooth_projection[::stride, 1],
        s=5,
        c="#9ecae1",
        alpha=0.48,
        linewidths=0,
    )

    if len(peak_projection):
        axis.scatter(
            peak_projection[:, 0],
            peak_projection[:, 1],
            s=26,
            c="#d62828",
            edgecolors="black",
            linewidths=0.28,
            label="peaks",
            zorder=5,
        )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("right projection")
    axis.set_ylabel("forward projection")
    axis.set_title(mesh_path.stem)
    axis.grid(alpha=0.15)
    if fallback_message:
        axis.text(
            0.02,
            0.02,
            "fallback: peaks only",
            transform=axis.transAxes,
            fontsize=8,
            color="#8d3b3b",
            ha="left",
            va="bottom",
        )
    if len(peak_projection):
        axis.legend(loc="best")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]

    # Edit these values directly for quick batch export.
    input_dir = root / "data" / "labeld_5year_betterv_objs"
    output_dir = root / "saved" / "LR"
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
        output_path = output_dir / f"{mesh_path.stem}_LR.png"
        if output_path.exists():
            print(f"[skip] {output_path.name} already exists")
            skipped_count += 1
            continue
        try:
            save_lr_peak_projection(
                mesh_path,
                output_path,
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
