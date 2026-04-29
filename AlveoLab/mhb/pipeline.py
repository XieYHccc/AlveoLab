from __future__ import annotations

import numpy as np

from AlveoLab.mhb.context import build_tooth_context
from AlveoLab.mhb.frame import GlobalFrame
from AlveoLab.mhb.labels import TOOTH_LABEL_DEFINITIONS
from AlveoLab.mhb.medial_curve import build_arch_medial_curve
from AlveoLab.mhb.models import ToothContext, ToothKeypointResult
from AlveoLab.mhb.orienter import ToothRegionPcaOrienter
from AlveoLab.mhb.registry import ToothRecognizerRegistry
from AlveoLab.mhb.utils import ensure_mesh


class MhbKeypointPipeline:
    def __init__(
        self,
        mesh,
        vertex_labels,
        *,
        arch_type: str | None = None,
        occlusal_axis: np.ndarray | None = None,
        orienter=None,
        registry: ToothRecognizerRegistry | None = None,
        keep_largest_component: bool = True,
    ):
        self.mesh = ensure_mesh(mesh)
        self.vertex_labels = np.asarray(vertex_labels, dtype=np.int64).reshape(-1)
        if self.vertex_labels.shape[0] != np.asarray(self.mesh.vertices).shape[0]:
            raise ValueError("vertex_labels length must match the mesh vertex count.")

        self.keep_largest_component = keep_largest_component
        self.registry = registry or ToothRecognizerRegistry()
        self.orienter = orienter or ToothRegionPcaOrienter(self.mesh, self.vertex_labels, arch_type or "L")
        self.global_frame = GlobalFrame.from_mesh(
            self.mesh,
            arch_type=arch_type,
            occlusal_axis=occlusal_axis,
            vertex_labels=self.vertex_labels,
            orienter=self.orienter,
        )
        self.medial_curve = build_arch_medial_curve(self.mesh, self.vertex_labels, self.global_frame)
        self.contexts = self._build_contexts()
        self.results = self.recognize()

    def _build_contexts(self) -> dict[int, ToothContext]:
        contexts: dict[int, ToothContext] = {}
        for label in sorted(TOOTH_LABEL_DEFINITIONS):
            context = build_tooth_context(
                self.mesh,
                self.vertex_labels,
                label,
                self.global_frame,
                medial_curve=self.medial_curve,
                keep_largest_component=self.keep_largest_component,
            )
            if context is not None:
                contexts[label] = context
        return contexts

    def recognize(self) -> dict[int, ToothKeypointResult]:
        outputs: dict[int, ToothKeypointResult] = {}
        for label, context in self.contexts.items():
            recognizer = self.registry.resolve(context.definition)
            outputs[label] = recognizer.recognize(context)
        return outputs


def recognize_mhb_keypoints(
    mesh,
    vertex_labels,
    *,
    arch_type: str | None = None,
    occlusal_axis: np.ndarray | None = None,
    orienter=None,
    registry: ToothRecognizerRegistry | None = None,
    keep_largest_component: bool = True,
) -> dict[int, ToothKeypointResult]:
    pipeline = MhbKeypointPipeline(
        mesh,
        vertex_labels,
        arch_type=arch_type,
        occlusal_axis=occlusal_axis,
        orienter=orienter,
        registry=registry,
        keep_largest_component=keep_largest_component,
    )
    return pipeline.results


if __name__ == "__main__":
    from pathlib import Path

    import pyvista as pv

    from AlveoLab.mesh import Mesh
    from AlveoLab.pyvista_utils import add_direction_frame, get_dental_plotter, mesh_to_polydata
    from AlveoLab.utils import infer_arch_type, load_labels
    root = Path(__file__).resolve().parents[2]

    # Edit these values directly for quick manual debugging.
    mesh_path = root / "data" / "labeld_5year_betterv_objs" / "0698_5 yr_Maxillary_export.obj"
    labels_path = root / "data" / "labeld_5year_betterv_objs" / "0698_5 yr_Maxillary_export.json"
    # labels_path = root / "saved" / "pred_labels_tgroupnet0302" / "0610_5yr_Maxillary_export.json"
    arch_type = infer_arch_type(mesh_path)
    show_medial_curve = False
    show_axes = False

    mesh = Mesh.from_file(mesh_path)
    vertex_labels = load_labels(labels_path, map=True)
    pipeline = MhbKeypointPipeline(mesh, vertex_labels, arch_type=arch_type)

    print(f"Mesh: {mesh_path}")
    print(f"Labels: {labels_path}")
    print(f"Arch type: {arch_type}")
    print(f"Teeth with contexts: {len(pipeline.contexts)}")
    print(f"Teeth with results: {len(pipeline.results)}")
    for label, result in sorted(pipeline.results.items()):
        keypoint_kinds = ", ".join(keypoint.kind for keypoint in result.keypoints) or "none"
        print(
            f"{label:>2} {result.tooth_name:<18} "
            f"{len(result.keypoints)} keypoints "
            f"[{result.recognizer_name}] {keypoint_kinds}"
        )

    plotter = get_dental_plotter()
    plotter.add_mesh(
        mesh_to_polydata(mesh),
        color=(1.0, 1.0, 1.0),
        opacity=1.0,
        specular=0.0,
        specular_power=5,
        ambient=0.2,
        show_edges=False,
    )

    tooth_points = np.asarray(mesh.vertices[vertex_labels > 0], dtype=float)
    tooth_height = float(np.median(tooth_points @ pipeline.global_frame.occlusal)) if tooth_points.size else 0.0
    if show_medial_curve and pipeline.medial_curve is not None:
        medial_points_3d = pipeline.medial_curve.points_2d_to_3d(
            pipeline.medial_curve.medial_points_2d,
            occlusal_offset=tooth_height,
        )
        plotter.add_mesh(pv.lines_from_points(medial_points_3d, close=False), color="#2a9d8f", line_width=4)

    def add_landmark_sphere(point: np.ndarray, color: str):
        plotter.add_mesh(
            pv.Sphere(radius=0.3, center=np.asarray(point, dtype=float)),
            color=color,
            opacity=0.8,
        )

    for label, result in sorted(pipeline.results.items()):
        if not result.keypoints:
            continue

        for keypoint in result.keypoints:
            point = np.asarray(keypoint.point, dtype=float)
            if "ridge" in keypoint.kind:
                add_landmark_sphere(point, color="red")
            else:
                add_landmark_sphere(point, color="green")

    if show_axes:
        axis_length = max(np.linalg.norm(np.ptp(np.asarray(mesh.vertices), axis=0)) * 0.12, 5.0)
        add_direction_frame(
            plotter,
            pipeline.global_frame.center,
            [
                ("right", pipeline.global_frame.right, "#d1495b"),
                ("forward", pipeline.global_frame.forward, "#00798c"),
                ("occlusal", pipeline.global_frame.occlusal, "#2a9d8f"),
            ],
            axis_length,
        )
    plotter.show()
