from __future__ import annotations

import numpy as np

from AlveoLab.mhb.cusp_detection import detect_cusps_local_extrema, detect_cusps_watershed
from AlveoLab.mhb.models import ToothContext, ToothKeypointResult
from AlveoLab.mhb.recognizers.base import BaseToothKeypointRecognizer


class CuspBasedMhbKeypointRecognizer(BaseToothKeypointRecognizer):
    name = "cusp_based"

    def __init__(
        self,
        *,
        max_candidates: int,
        method: str = "watershed",
        height_function_mode: str = "height_only",
        occlusal_quantile: float = 0.8,
        alpha: float = 0.5,
        min_basin_depth: float | None = None,
        min_basin_depth_ratio: float | None = None,
        min_basin_size: int = 3,
        flat_tolerance: float = 1e-8,
        include_debug_arrays: bool = False,
        min_separation_ratio: float = 0.18,
        min_separation_mm: float = 0.8,
    ):
        if method not in {"watershed", "local_extrema"}:
            raise ValueError("method must be either 'watershed' or 'local_extrema'.")
        if height_function_mode not in {"height_only", "combined"}:
            raise ValueError("height_function_mode must be either 'height_only' or 'combined'.")

        self.max_candidates = int(max_candidates)
        self.method = method
        self.height_function_mode = height_function_mode
        self.occlusal_quantile = float(occlusal_quantile)
        self.alpha = float(alpha)
        self.min_basin_depth = min_basin_depth
        self.min_basin_depth_ratio = (
            None if min_basin_depth_ratio is None else float(min_basin_depth_ratio)
        )
        self.min_basin_size = int(min_basin_size)
        self.flat_tolerance = float(flat_tolerance)
        self.include_debug_arrays = bool(include_debug_arrays)
        self.min_separation_ratio = float(min_separation_ratio)
        self.min_separation_mm = float(min_separation_mm)

    def keypoint_kind(self, rank: int) -> str:
        return f"cusp_{rank + 1}"

    def recognize(self, context: ToothContext) -> ToothKeypointResult:
        if self.method == "watershed":
            detection = detect_cusps_watershed(
                context,
                height_function_mode=self.height_function_mode,
                alpha=self.alpha,
                max_candidates=self.max_candidates,
                min_basin_depth=self.min_basin_depth,
                min_basin_depth_ratio=self.min_basin_depth_ratio,
                min_basin_size=self.min_basin_size,
                flat_tolerance=self.flat_tolerance,
                include_debug_arrays=self.include_debug_arrays,
            )
        else:
            detection = detect_cusps_local_extrema(
                context,
                max_candidates=self.max_candidates,
                occlusal_quantile=self.occlusal_quantile,
                min_separation_ratio=self.min_separation_ratio,
                min_separation_mm=self.min_separation_mm,
            )

        if not detection.candidates:
            raise RuntimeError("Cusp detection produced no candidates.")

        keypoints = [
            self._make_keypoint(
                context,
                candidate.local_index,
                kind=self.keypoint_kind(rank),
                score=candidate.score,
                basin_size=candidate.basin_size,
                basin_depth=candidate.basin_depth,
                height_value=candidate.height_value,
                mean_curvature=candidate.curvature_value,
                elevation_value=candidate.elevation_value,
                peak_elevation_value=candidate.peak_elevation_value,
                detection_method=self.method,
            )
            for rank, candidate in enumerate(detection.candidates)
        ]

        return self._result(
            context,
            keypoints,
            max_candidates=self.max_candidates,
            detection_method=self.method,
            occlusal_quantile=self.occlusal_quantile,
            **detection.debug,
        )


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np
    import pyvista as pv

    from AlveoLab.mesh import Mesh
    from AlveoLab.mhb.labels import CANINE, INCISOR, MOLAR, PRIMARY_MOLAR
    from AlveoLab.mhb.pipeline import MhbKeypointPipeline
    from AlveoLab.pyvista_utils import get_dental_plotter, mesh_to_polydata
    from AlveoLab.utils import infer_arch_type, load_labels

    root = Path(__file__).resolve().parents[3]

    # Edit these values directly for quick manual debugging.
    mesh_path = root / "data" / "labeld_5year_betterv_objs" / "VAL1_UpperJaw_070819.obj"
    labels_path = root / "data" / "labeld_5year_betterv_objs" / "VAL1_UpperJaw_070819.json"
    arch_type = infer_arch_type(mesh_path)
    map_labels = True
    method = "watershed"
    height_function_mode = "height_only"
    max_candidates = 5
    include_incisors = True
    apply_family_postfilter = True

    show_full_mesh_after = True

    if method != "watershed":
        raise RuntimeError("This debug view requires watershed mode because basin visualization is requested.")

    mesh = Mesh.from_file(mesh_path)
    vertex_labels = load_labels(labels_path, map=map_labels)
    pipeline = MhbKeypointPipeline(mesh, vertex_labels, arch_type=arch_type)

    basin_color_table = [
        np.array([0.470, 0.700, 0.955], dtype=float),  # light blue
        np.array([0.500, 0.860, 0.590], dtype=float),  # light green
        np.array([0.975, 0.885, 0.430], dtype=float),  # light yellow
        np.array([0.520, 0.885, 0.955], dtype=float),  # light cyan
        np.array([0.885, 0.580, 0.820], dtype=float),  # light magenta
        np.array([0.990, 0.710, 0.430], dtype=float),  # light orange
        np.array([0.460, 0.830, 0.790], dtype=float),  # light teal
        np.array([0.730, 0.660, 0.975], dtype=float),  # light violet
        np.array([0.895, 0.810, 0.470], dtype=float),  # light olive
        np.array([0.430, 0.780, 0.975], dtype=float),  # sky blue
        np.array([0.620, 0.910, 0.660], dtype=float),  # spring green
        np.array([0.995, 0.790, 0.560], dtype=float),  # soft amber
    ]

    def basin_color(index: int) -> np.ndarray:
        base_color = np.asarray(basin_color_table[index % len(basin_color_table)], dtype=float)
        cycle = index // len(basin_color_table)
        if cycle <= 0:
            return base_color
        blend = min(0.22 * cycle, 0.45)
        return np.clip((1.0 - blend) * base_color + blend * np.array([1.0, 1.0, 1.0], dtype=float), 0.0, 1.0)

    def add_cusp_spheres(plotter, points: np.ndarray, *, color: str = "red", radius: float = 0.3, opacity: float = 0.8):
        points = np.asarray(points, dtype=float)
        for point in points:
            plotter.add_mesh(
                pv.Sphere(radius=radius, center=point),
                color=color,
                opacity=opacity,
            )

    def make_debug_recognizer(family: str) -> CuspBasedMhbKeypointRecognizer:
        if not apply_family_postfilter:
            return CuspBasedMhbKeypointRecognizer(
                max_candidates=(1 if family == CANINE else max_candidates),
                method=method,
                height_function_mode=height_function_mode,
                include_debug_arrays=True,
            )

        if family == CANINE:
            from AlveoLab.mhb.recognizers.canine import CanineMhbKeypointRecognizer

            return CanineMhbKeypointRecognizer(
                method=method,
                height_function_mode=height_function_mode,
                include_debug_arrays=True,
            )
        if family == PRIMARY_MOLAR:
            from AlveoLab.mhb.recognizers.primary_molar import PrimaryMolarMhbKeypointRecognizer

            return PrimaryMolarMhbKeypointRecognizer(
                max_candidates=max_candidates,
                method=method,
                height_function_mode=height_function_mode,
                include_debug_arrays=True,
            )
        if family == MOLAR:
            from AlveoLab.mhb.recognizers.molar import MolarMhbKeypointRecognizer

            return MolarMhbKeypointRecognizer(
                max_candidates=max_candidates,
                method=method,
                height_function_mode=height_function_mode,
                include_debug_arrays=True,
            )
        raise KeyError(f"Unsupported tooth family: {family}")

    target_contexts = []
    incisor_meshes = []
    for label, context in sorted(pipeline.contexts.items()):
        if context.definition.family == INCISOR:
            if include_incisors:
                selected_vertex_mask = np.asarray(vertex_labels, dtype=np.int64) == int(label)
                tooth_face_mask = np.all(selected_vertex_mask[np.asarray(mesh.faces)], axis=1)
                if not np.any(tooth_face_mask):
                    tooth_face_mask = selected_vertex_mask[np.asarray(mesh.faces)].sum(axis=1) >= 2
                tooth_faces_global = np.asarray(mesh.faces[tooth_face_mask], dtype=np.int64)
                if tooth_faces_global.size:
                    unique_global_vertices, inverse = np.unique(tooth_faces_global.reshape(-1), return_inverse=True)
                    incisor_meshes.append(
                        Mesh.from_vertices_faces(
                            np.asarray(mesh.vertices[unique_global_vertices], dtype=float),
                            inverse.reshape(-1, 3),
                        )
                    )
            continue
        if context.definition.family not in {CANINE, PRIMARY_MOLAR, MOLAR}:
            continue
        target_contexts.append((label, context))

    if not target_contexts:
        raise RuntimeError("No non-incisor tooth contexts were found for basin visualization.")

    debug_results = []
    global_basin_counter = 0
    for label, context in target_contexts:
        recognizer = make_debug_recognizer(context.definition.family)
        result = recognizer.recognize(context)
        final_basin_labels = np.asarray(result.debug.get("final_basin_labels"), dtype=np.int64)
        if final_basin_labels.size == 0:
            continue

        global_to_context_local = {
            int(vertex_index): local_index
            for local_index, vertex_index in enumerate(np.asarray(context.vertex_indices, dtype=np.int64))
        }
        detected_local_indices = np.asarray(
            [
                int(global_to_context_local[int(keypoint.vertex_index)])
                for keypoint in result.keypoints
                if int(keypoint.vertex_index) in global_to_context_local
            ],
            dtype=np.int64,
        )
        detected_basin_minima = sorted(
            {
                int(final_basin_labels[int(local_index)])
                for local_index in detected_local_indices
            }
        )
        if not detected_basin_minima:
            continue
        merged_basin_minima = sorted(
            int(local_index)
            for local_index in np.asarray(result.debug.get("merged_basin_minima_local_indices"), dtype=np.int64)
        )
        rejected_basin_minima = sorted(set(merged_basin_minima) - set(detected_basin_minima))

        selected_vertex_mask = np.asarray(vertex_labels, dtype=np.int64) == int(label)
        tooth_face_mask = np.all(selected_vertex_mask[np.asarray(mesh.faces)], axis=1)
        if not np.any(tooth_face_mask):
            tooth_face_mask = selected_vertex_mask[np.asarray(mesh.faces)].sum(axis=1) >= 2
        tooth_faces_global = np.asarray(mesh.faces[tooth_face_mask], dtype=np.int64)
        if tooth_faces_global.size == 0:
            continue

        unique_global_vertices, inverse = np.unique(tooth_faces_global.reshape(-1), return_inverse=True)
        tooth_vertices = np.asarray(mesh.vertices[unique_global_vertices], dtype=float)
        tooth_faces_local = inverse.reshape(-1, 3)
        tooth_mesh = Mesh.from_vertices_faces(tooth_vertices, tooth_faces_local)

        basin_palette = {}
        for basin_minimum in detected_basin_minima:
            color = basin_color(global_basin_counter)
            basin_palette[int(basin_minimum)] = color
            global_basin_counter += 1

        rejected_basin_color = np.array([0.86, 0.86, 0.86], dtype=float)
        default_color = np.array([1.0, 1.0, 1.0], dtype=float)
        tooth_vertex_colors = np.tile(default_color, (unique_global_vertices.shape[0], 1))
        for vertex_row, global_vertex_index in enumerate(unique_global_vertices):
            context_local_index = global_to_context_local.get(int(global_vertex_index))
            if context_local_index is None:
                continue
            basin_minimum = int(final_basin_labels[context_local_index])
            if basin_minimum in basin_palette:
                tooth_vertex_colors[vertex_row] = basin_palette[basin_minimum]
            elif basin_minimum in rejected_basin_minima:
                tooth_vertex_colors[vertex_row] = rejected_basin_color

        debug_results.append(
            {
                "label": int(label),
                "context": context,
                "result": result,
                "detected_basin_minima": detected_basin_minima,
                "rejected_basin_minima": rejected_basin_minima,
                "tooth_mesh": tooth_mesh,
                "tooth_vertex_colors": tooth_vertex_colors,
            }
        )

    cusp_points = np.asarray(
        [keypoint.point for item in debug_results for keypoint in item["result"].keypoints],
        dtype=float,
    )
    if cusp_points.size == 0:
        raise RuntimeError("No cusp points were detected for the selected non-incisor teeth.")

    print(f"Mesh: {mesh_path}")
    print(f"Labels: {labels_path}")
    print(f"Arch type: {arch_type}")
    print(f"Method: {method}")
    print(f"Height function mode: {height_function_mode}")
    print(f"Apply family post-filter: {apply_family_postfilter}")
    print(f"Show incisors without keypoints: {include_incisors}")
    print(f"Visualized teeth: {[item['label'] for item in debug_results]}")
    print(f"Detected cusps: {sum(len(item['result'].keypoints) for item in debug_results)}")
    for item in debug_results:
        context = item["context"]
        result = item["result"]
        print(
            f"Tooth {item['label']} ({context.definition.name}): "
            f"{len(result.keypoints)} cusps, basins={item['detected_basin_minima']}, "
            f"rejected_basins={item['rejected_basin_minima']}"
        )

    plotter = get_dental_plotter()
    for incisor_mesh in incisor_meshes:
        plotter.add_mesh(
            mesh_to_polydata(incisor_mesh),
            color=(1.0, 1.0, 1.0),
            opacity=1.0,
            specular=0.0,
            specular_power=5,
            ambient=0.2,
            show_edges=False,
        )
    for item in debug_results:
        pv_tooth = mesh_to_polydata(item["tooth_mesh"])
        pv_tooth["basin_colors"] = item["tooth_vertex_colors"]
        plotter.add_mesh(
            pv_tooth,
            scalars="basin_colors",
            rgb=True,
            opacity=1.0,
            specular=0.0,
            specular_power=5,
            ambient=0.2,
            show_edges=False,
        )
    plotter.show()

    if show_full_mesh_after:
        full_plotter = get_dental_plotter()
        full_plotter.add_mesh(
            mesh_to_polydata(mesh),
            color=(1.0, 1.0, 1.0),
            opacity=1.0,
            specular=0.0,
            specular_power=5,
            ambient=0.2,
            show_edges=False,
        )
        add_cusp_spheres(full_plotter, cusp_points, color="red", radius=0.3, opacity=0.8)
        full_plotter.show()
