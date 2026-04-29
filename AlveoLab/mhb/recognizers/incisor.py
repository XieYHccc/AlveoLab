from __future__ import annotations

import numpy as np

from AlveoLab.mhb.cross_section import build_tooth_cross_sections, contiguous_true_runs, normalized_polyline_curvature
from AlveoLab.mhb.models import ToothContext, ToothKeypointResult
from AlveoLab.mhb.recognizers.base import BaseToothKeypointRecognizer
from AlveoLab.mhb.utils import clip_percentile_range, scale_to_unit_interval


class IncisorMhbKeypointRecognizer(BaseToothKeypointRecognizer):
    name = "incisor"

    def __init__(
        self,
        curvature_quantile: float = 0.9,
        min_ridge_vertices: int = 12,
        *,
        use_cross_section_ridge: bool = False,
        cross_section_count: int = 11,
        cross_section_margin_ratio: float = 0.08,
        cross_section_curvature_threshold: float = 0.6,
        cross_section_connection_distance: float = 0.75,
        cross_section_noise_ratio: float = 0.1,
        cross_section_top_quantile: float = 0.7,
        occlusal_quantile: float | None = None,
        min_band_vertices: int | None = None,
    ):
        if occlusal_quantile is not None:
            curvature_quantile = occlusal_quantile
        if min_band_vertices is not None:
            min_ridge_vertices = min_band_vertices
        self.curvature_quantile = curvature_quantile
        self.min_ridge_vertices = min_ridge_vertices
        self.use_cross_section_ridge = bool(use_cross_section_ridge)
        self.cross_section_count = int(cross_section_count)
        self.cross_section_margin_ratio = float(cross_section_margin_ratio)
        self.cross_section_curvature_threshold = float(cross_section_curvature_threshold)
        self.cross_section_connection_distance = float(cross_section_connection_distance)
        self.cross_section_noise_ratio = float(cross_section_noise_ratio)
        self.cross_section_top_quantile = float(cross_section_top_quantile)

    def _ridge_curvature(self, context: ToothContext) -> tuple[np.ndarray, float, float]:
        mean_curvature = np.asarray(context.mesh.vertex_mean_curvature[context.vertex_indices], dtype=float)
        clipped_curvature, curvature_clip_min, curvature_clip_max = clip_percentile_range(mean_curvature, 5.0, 95.0)
        positive_curvature = np.clip(clipped_curvature, a_min=0.0, a_max=None)
        return positive_curvature, curvature_clip_min, curvature_clip_max

    def _fallback_occlusal_candidates(self, context: ToothContext) -> np.ndarray:
        occlusal_values = context.occlusal_values
        take = min(self.min_ridge_vertices, occlusal_values.size)
        top_order = np.argsort(occlusal_values)[-take:]
        candidate_mask = np.zeros_like(occlusal_values, dtype=bool)
        candidate_mask[top_order] = True
        inner_candidate_mask = candidate_mask & ~context.boundary_mask
        if np.any(inner_candidate_mask):
            candidate_mask = inner_candidate_mask
        return np.flatnonzero(candidate_mask)

    def _estimate_incisal_direction(self, context: ToothContext, positive_curvature: np.ndarray) -> np.ndarray:
        md = context.mesiodistal_values
        bl = context.buccolingual_values
        occlusal = context.occlusal_values
        md_center = float(np.median(md))
        md_range = max(float(np.ptp(md)), 1e-6)
        central_weights = np.clip(1.0 - ((md - md_center) / md_range) ** 2, a_min=0.0, a_max=None)
        profile_points = np.column_stack((bl, occlusal))
        tooth_center_2d = profile_points.mean(axis=0)
        ridge_weights = positive_curvature * central_weights

        if np.any(ridge_weights > 0):
            ridge_center_2d = np.average(profile_points, axis=0, weights=ridge_weights)
            incisal_direction_2d = ridge_center_2d - tooth_center_2d
        else:
            incisal_direction_2d = np.array([0.0, 1.0], dtype=float)

        norm = float(np.linalg.norm(incisal_direction_2d))
        if norm < 1e-8:
            incisal_direction_2d = np.array([0.0, 1.0], dtype=float)
        else:
            incisal_direction_2d = incisal_direction_2d / norm
        if incisal_direction_2d[1] < 0:
            incisal_direction_2d = -incisal_direction_2d
        return incisal_direction_2d

    def _nearest_local_index(self, context: ToothContext, point_3d: np.ndarray) -> int:
        distances = np.linalg.norm(np.asarray(context.points, dtype=float) - point_3d[np.newaxis, :], axis=1)
        return int(np.argmin(distances))

    def _select_incisal_curve_segment(self, curve_2d: np.ndarray) -> tuple[np.ndarray, int, float, np.ndarray]:
        normalized_curvature = normalized_polyline_curvature(curve_2d, smoothing_window=5)
        occlusal_values = curve_2d[:, 1]
        top_threshold = float(np.quantile(occlusal_values, self.cross_section_top_quantile))
        top_mask = occlusal_values >= top_threshold
        if np.any(top_mask):
            if float(np.max(-normalized_curvature[top_mask])) > float(np.max(normalized_curvature[top_mask])):
                normalized_curvature = -normalized_curvature

        candidate_mask = top_mask & (normalized_curvature >= self.cross_section_curvature_threshold)
        runs = contiguous_true_runs(candidate_mask)
        if not runs and np.any(top_mask):
            relaxed_threshold = max(0.25, 0.5 * self.cross_section_curvature_threshold)
            runs = contiguous_true_runs(top_mask & (normalized_curvature >= relaxed_threshold))
        if not runs:
            return np.empty(0, dtype=np.int64), -1, 0.0, normalized_curvature

        occlusal_unit = scale_to_unit_interval(occlusal_values)
        best_indices = np.empty(0, dtype=np.int64)
        best_rep = -1
        best_score = -np.inf
        for start, end in runs:
            run_indices = np.arange(start, end + 1, dtype=np.int64)
            run_score = 0.65 * float(np.mean(normalized_curvature[run_indices])) + 0.35 * float(
                np.mean(occlusal_unit[run_indices])
            )
            point_scores = 0.7 * normalized_curvature[run_indices] + 0.3 * occlusal_unit[run_indices]
            rep_index = int(run_indices[int(np.argmax(point_scores))])
            if run_score > best_score:
                best_indices = run_indices
                best_rep = rep_index
                best_score = run_score
        return best_indices, best_rep, float(best_score), normalized_curvature

    def _extract_cross_section_ridge(self, context: ToothContext) -> dict[str, object]:
        sections = build_tooth_cross_sections(
            context,
            orientation="buccolingual",
            count=self.cross_section_count,
            margin_ratio=self.cross_section_margin_ratio,
            min_curve_points=5,
        )
        if not sections:
            return {
                "success": False,
                "failure_reason": "no_sections",
                "ridge_indices": np.empty(0, dtype=np.int64),
                "ridge_scores": np.empty(0, dtype=float),
                "ridge_path_points": np.empty((0, 3), dtype=float),
                "candidate_sections": 0,
                "support": 0,
                "total_sections": 0,
                "section_debug": [],
            }

        candidates: list[dict[str, object]] = []
        section_debug: list[dict[str, object]] = []
        for section in sections:
            curve_2d = np.asarray(section.curve_2d, dtype=float)
            selected_curve_indices = np.empty(0, dtype=np.int64)
            debug_status = str(section.status)
            representative_curve_index = -1
            normalized_curvature = np.zeros(curve_2d.shape[0], dtype=float)

            if section.status != "ok" or curve_2d.shape[0] < 5:
                section_debug.append(
                    {
                        "section_index": int(section.section_index),
                        "status": debug_status,
                        "curve_3d": np.asarray(section.curve_3d, dtype=float),
                        "curve_2d": curve_2d,
                        "normalized_curvature": normalized_curvature,
                        "selected_curve_indices": selected_curve_indices,
                        "representative_curve_index": representative_curve_index,
                        "stitched": False,
                    }
                )
                continue

            selected_curve_indices, representative_curve_index, score, normalized_curvature = self._select_incisal_curve_segment(curve_2d)
            if representative_curve_index < 0 or selected_curve_indices.size == 0:
                section_debug.append(
                    {
                        "section_index": int(section.section_index),
                        "status": "no_selected_segment",
                        "curve_3d": np.asarray(section.curve_3d, dtype=float),
                        "curve_2d": curve_2d,
                        "normalized_curvature": normalized_curvature,
                        "selected_curve_indices": selected_curve_indices,
                        "representative_curve_index": representative_curve_index,
                        "stitched": False,
                    }
                )
                continue

            section_debug.append(
                {
                    "section_index": int(section.section_index),
                    "status": "selected",
                    "curve_3d": np.asarray(section.curve_3d, dtype=float),
                    "curve_2d": curve_2d,
                    "normalized_curvature": normalized_curvature,
                    "selected_curve_indices": selected_curve_indices,
                    "representative_curve_index": representative_curve_index,
                    "stitched": False,
                }
            )
            representative_point = np.asarray(section.curve_3d[representative_curve_index], dtype=float)
            candidates.append(
                {
                    "section_index": int(section.section_index),
                    "local_index": self._nearest_local_index(context, representative_point),
                    "point_3d": representative_point,
                    "score": float(score),
                }
            )

        if not candidates:
            return {
                "success": False,
                "failure_reason": "no_section_candidates",
                "ridge_indices": np.empty(0, dtype=np.int64),
                "ridge_scores": np.empty(0, dtype=float),
                "ridge_path_points": np.empty((0, 3), dtype=float),
                "candidate_sections": 0,
                "support": 0,
                "total_sections": len(sections),
                "section_debug": section_debug,
            }

        candidates = sorted(candidates, key=lambda candidate: int(candidate["section_index"]))
        allowed_gap = max(1, int(round(self.cross_section_noise_ratio * max(len(sections), 1))))
        runs: list[list[dict[str, object]]] = []
        current_run = [candidates[0]]
        for candidate in candidates[1:]:
            previous = current_run[-1]
            gap = int(candidate["section_index"]) - int(previous["section_index"]) - 1
            distance = float(np.linalg.norm(np.asarray(candidate["point_3d"]) - np.asarray(previous["point_3d"])))
            if gap <= allowed_gap and distance <= self.cross_section_connection_distance * float(gap + 1):
                current_run.append(candidate)
            else:
                runs.append(current_run)
                current_run = [candidate]
        runs.append(current_run)

        best_run = max(runs, key=lambda run: (len(run), float(sum(float(item["score"]) for item in run))))
        min_support = max(2, int(np.ceil(0.2 * max(len(sections), 1))))
        if len(best_run) < min_support:
            return {
                "success": False,
                "failure_reason": "stitching_failed",
                "ridge_indices": np.empty(0, dtype=np.int64),
                "ridge_scores": np.empty(0, dtype=float),
                "ridge_path_points": np.empty((0, 3), dtype=float),
                "candidate_sections": len(candidates),
                "support": 0,
                "total_sections": len(sections),
                "section_debug": section_debug,
            }

        ridge_indices: list[int] = []
        ridge_scores: list[float] = []
        ridge_path_points: list[np.ndarray] = []
        stitched_ids = {int(item["section_index"]) for item in best_run}
        for item in section_debug:
            item["stitched"] = int(item["section_index"]) in stitched_ids
        for item in best_run:
            local_index = int(item["local_index"])
            if ridge_indices and ridge_indices[-1] == local_index:
                continue
            ridge_indices.append(local_index)
            ridge_scores.append(float(item["score"]))
            ridge_path_points.append(np.asarray(item["point_3d"], dtype=float))

        return {
            "success": True,
            "failure_reason": "",
            "ridge_indices": np.asarray(ridge_indices, dtype=np.int64),
            "ridge_scores": np.asarray(ridge_scores, dtype=float),
            "ridge_path_points": np.asarray(ridge_path_points, dtype=float),
            "candidate_sections": len(candidates),
            "support": len(best_run),
            "total_sections": len(sections),
            "section_debug": section_debug,
        }

    def _trace_geometric_ridge(
        self,
        context: ToothContext,
        positive_curvature: np.ndarray,
        incisal_direction_2d: np.ndarray,
    ) -> dict[str, np.ndarray | float | str]:
        md = context.mesiodistal_values
        bl = context.buccolingual_values
        occlusal = context.occlusal_values
        incisal_projection = bl * incisal_direction_2d[0] + occlusal * incisal_direction_2d[1]

        usable_mask = ~context.boundary_mask
        usable_count = int(np.count_nonzero(usable_mask))
        num_bins = int(np.clip(np.sqrt(max(usable_count, 1)), 5, 17))
        bin_edges = np.linspace(float(md.min()), float(md.max()), num_bins + 1)

        top_envelope_mask = np.zeros_like(usable_mask, dtype=bool)
        ridge_indices: list[int] = []
        ridge_scores: list[float] = []
        envelope_tolerances: list[float] = []

        usable_projection = incisal_projection[usable_mask]
        global_projection_range = max(float(np.ptp(usable_projection)) if usable_projection.size else 0.0, 1e-6)

        for bin_index in range(num_bins):
            left = bin_edges[bin_index]
            right = bin_edges[bin_index + 1]
            in_bin = usable_mask & (md >= left)
            if bin_index < num_bins - 1:
                in_bin &= md < right
            else:
                in_bin &= md <= right
            if not np.any(in_bin):
                continue

            bin_projection = incisal_projection[in_bin]
            bin_max_projection = float(bin_projection.max())
            local_projection_range = float(np.ptp(bin_projection)) if bin_projection.size > 1 else 0.0
            envelope_tolerance = max(0.08 * global_projection_range, 0.15 * local_projection_range, 1e-6)
            envelope_tolerances.append(envelope_tolerance)

            top_mask_bin = in_bin & (incisal_projection >= bin_max_projection - envelope_tolerance)
            top_envelope_mask |= top_mask_bin
            top_indices = np.flatnonzero(top_mask_bin)
            if top_indices.size == 0:
                continue

            envelope_scores = 1.0 - np.clip(
                (bin_max_projection - incisal_projection[top_indices]) / envelope_tolerance,
                a_min=0.0,
                a_max=1.0,
            )
            local_scores = (
                0.45 * scale_to_unit_interval(positive_curvature[top_indices])
                + 0.45 * envelope_scores
                + 0.10 * scale_to_unit_interval(occlusal[top_indices])
            )
            ridge_indices.append(int(top_indices[int(np.argmax(local_scores))]))
            ridge_scores.append(float(np.max(local_scores)))

        if not ridge_indices:
            fallback_indices = self._fallback_occlusal_candidates(context)
            ridge_indices = [int(fallback_indices[0])] if fallback_indices.size else [int(np.argmax(context.occlusal_values))]
            ridge_scores = [1.0]
            selection_source = "occlusal_fallback"
        else:
            selection_source = "mean_curvature_geometric_ridge"

        deduplicated_scores: dict[int, float] = {}
        for ridge_index, ridge_score in zip(ridge_indices, ridge_scores, strict=False):
            deduplicated_scores.setdefault(int(ridge_index), float(ridge_score))
        ordered_ridge_indices = sorted(deduplicated_scores, key=lambda index: md[index])
        ridge_indices_array = np.asarray(ordered_ridge_indices, dtype=np.int64)
        ridge_scores_array = np.asarray([deduplicated_scores[index] for index in ordered_ridge_indices], dtype=float)
        return {
            "ridge_indices": ridge_indices_array,
            "ridge_scores": ridge_scores_array,
            "ridge_path_points": np.asarray(context.points[ridge_indices_array], dtype=float),
            "top_envelope_mask": top_envelope_mask,
            "positive_curvature": positive_curvature,
            "incisal_direction_2d": incisal_direction_2d,
            "envelope_tolerance": float(np.median(envelope_tolerances)) if envelope_tolerances else 0.0,
            "selection_source": selection_source,
        }

    def _midpoint_from_ridge_path(
        self,
        context: ToothContext,
        ridge_indices: np.ndarray,
        ridge_path_points: np.ndarray,
    ) -> tuple[int, int]:
        if ridge_indices.size == 0:
            fallback_index = int(np.argmax(context.occlusal_values))
            return fallback_index, 0
        if ridge_path_points.shape[0] <= 1:
            return int(ridge_indices[0]), 0

        segment_lengths = np.linalg.norm(np.diff(ridge_path_points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        midpoint_rank = int(np.argmin(np.abs(cumulative - 0.5 * cumulative[-1])))
        midpoint_point = np.asarray(ridge_path_points[midpoint_rank], dtype=float)
        return self._nearest_local_index(context, midpoint_point), midpoint_rank

    def _ridge_debug(self, context: ToothContext) -> dict[str, object]:
        positive_curvature, curvature_clip_min, curvature_clip_max = self._ridge_curvature(context)
        incisal_direction_2d = self._estimate_incisal_direction(context, positive_curvature)

        if self.use_cross_section_ridge:
            cross_section = self._extract_cross_section_ridge(context)
        else:
            cross_section = {
                "success": False,
                "failure_reason": "disabled",
                "ridge_indices": np.empty(0, dtype=np.int64),
                "ridge_scores": np.empty(0, dtype=float),
                "ridge_path_points": np.empty((0, 3), dtype=float),
                "candidate_sections": 0,
                "support": 0,
                "total_sections": 0,
                "section_debug": [],
            }

        if self.use_cross_section_ridge and bool(cross_section["success"]):
            ridge_indices = np.asarray(cross_section["ridge_indices"], dtype=np.int64)
            ridge_path_points = np.asarray(cross_section["ridge_path_points"], dtype=float)
            midpoint_index, midpoint_rank = self._midpoint_from_ridge_path(context, ridge_indices, ridge_path_points)
            return {
                "ridge_indices": ridge_indices,
                "ridge_scores": np.asarray(cross_section["ridge_scores"], dtype=float),
                "ridge_path_points": ridge_path_points,
                "top_envelope_mask": np.zeros_like(context.boundary_mask, dtype=bool),
                "positive_curvature": positive_curvature,
                "incisal_direction_2d": incisal_direction_2d,
                "envelope_tolerance": 0.0,
                "selection_source": "cross_section_ridge",
                "curvature_clip_min": curvature_clip_min,
                "curvature_clip_max": curvature_clip_max,
                "midpoint_index": midpoint_index,
                "midpoint_rank": midpoint_rank,
                "cross_section_enabled": True,
                "cross_section_success": True,
                "cross_section_support": int(cross_section["support"]),
                "cross_section_candidate_sections": int(cross_section["candidate_sections"]),
                "cross_section_total_sections": int(cross_section["total_sections"]),
                "cross_section_failure_reason": "",
                "cross_section_debug": cross_section["section_debug"],
            }

        ridge = self._trace_geometric_ridge(context, positive_curvature, incisal_direction_2d)
        ridge_indices = np.asarray(ridge["ridge_indices"], dtype=np.int64)
        ridge_path_points = np.asarray(ridge["ridge_path_points"], dtype=float)
        midpoint_index, midpoint_rank = self._midpoint_from_ridge_path(context, ridge_indices, ridge_path_points)
        return {
            **ridge,
            "curvature_clip_min": curvature_clip_min,
            "curvature_clip_max": curvature_clip_max,
            "midpoint_index": midpoint_index,
            "midpoint_rank": midpoint_rank,
            "cross_section_enabled": self.use_cross_section_ridge,
            "cross_section_success": False,
            "cross_section_support": 0,
            "cross_section_candidate_sections": int(cross_section["candidate_sections"]),
            "cross_section_total_sections": int(cross_section["total_sections"]),
            "cross_section_failure_reason": str(cross_section["failure_reason"]),
            "cross_section_debug": cross_section["section_debug"],
        }

    def recognize(self, context: ToothContext) -> ToothKeypointResult:
        debug = self._ridge_debug(context)
        ridge_indices = np.asarray(debug["ridge_indices"], dtype=np.int64)
        ridge_scores = np.asarray(debug["ridge_scores"], dtype=float)
        midpoint_index = int(debug["midpoint_index"])
        midpoint_rank = int(debug["midpoint_rank"])
        midpoint_score = float(ridge_scores[min(midpoint_rank, max(ridge_scores.size - 1, 0))]) if ridge_scores.size else 1.0
        positive_curvature = np.asarray(debug["positive_curvature"], dtype=float)

        keypoint = self._make_keypoint(
            context,
            midpoint_index,
            kind="ridge_midpoint",
            score=midpoint_score,
            mean_curvature=float(positive_curvature[midpoint_index]),
            ridge_point_count=int(ridge_indices.size),
            selection_source=str(debug["selection_source"]),
        )
        return self._result(
            context,
            [keypoint],
            curvature_source="vertex_mean_curvature",
            curvature_sign_constraint="positive_only",
            curvature_clip_percentiles=(5.0, 95.0),
            curvature_clip_min=float(debug["curvature_clip_min"]),
            curvature_clip_max=float(debug["curvature_clip_max"]),
            ridge_point_count=int(ridge_indices.size),
            envelope_tolerance=float(debug["envelope_tolerance"]),
            incisal_direction_2d=np.asarray(debug["incisal_direction_2d"], dtype=float).tolist(),
            selection_source=str(debug["selection_source"]),
            cross_section_enabled=bool(debug["cross_section_enabled"]),
            cross_section_success=bool(debug["cross_section_success"]),
            cross_section_support=int(debug["cross_section_support"]),
            cross_section_candidate_sections=int(debug["cross_section_candidate_sections"]),
            cross_section_total_sections=int(debug["cross_section_total_sections"]),
            cross_section_failure_reason=str(debug["cross_section_failure_reason"]),
            cross_section_debug=debug["cross_section_debug"],
        )


if __name__ == "__main__":
    from pathlib import Path

    import pyvista as pv

    from AlveoLab.mesh import Mesh
    from AlveoLab.mhb.pipeline import MhbKeypointPipeline
    from AlveoLab.pyvista_utils import get_dental_plotter, mesh_to_polydata
    from AlveoLab.utils import infer_arch_type, load_labels

    root = Path(__file__).resolve().parents[3]

    # Edit these values directly for quick manual debugging / thesis screenshots.
    mesh_path = root / "data" / "labeld_5year_betterv_objs" / "0609_5 YR_Maxillary_export.obj"
    labels_path = root / "data" / "labeld_5year_betterv_objs" / "0609_5 YR_Maxillary_export.json"
    arch_type = infer_arch_type(mesh_path)
    map_labels = True
    use_cross_section_ridge = False

    mesh = Mesh.from_file(mesh_path)
    vertex_labels = load_labels(labels_path, map=map_labels)
    pipeline = MhbKeypointPipeline(mesh, vertex_labels, arch_type=arch_type)

    recognizer = IncisorMhbKeypointRecognizer(use_cross_section_ridge=use_cross_section_ridge)
    incisor_labels = [
        label
        for label, context in sorted(pipeline.contexts.items())
        if context.definition.family == "incisor"
    ]

    if not incisor_labels:
        raise ValueError("No incisor contexts were found in the current mesh/labels.")

    incisor_debug_data: list[tuple[ToothContext, dict[str, object], ToothKeypointResult]] = []
    full_vertex_curvature = np.zeros(np.asarray(mesh.vertices).shape[0], dtype=float)

    print(f"Mesh: {mesh_path}")
    print(f"Labels: {labels_path}")
    print(f"Arch type: {arch_type}")
    print(f"Incisor labels: {incisor_labels}")

    def add_midpoint_sphere(plotter, point: np.ndarray, *, color: str = "red", radius: float = 0.3, opacity: float = 0.8):
        plotter.add_mesh(
            pv.Sphere(radius=radius, center=np.asarray(point, dtype=float)),
            color=color,
            opacity=opacity,
        )

    def add_full_mesh(plotter):
        plotter.add_mesh(
            mesh_to_polydata(mesh),
            color=(1.0, 1.0, 1.0),
            opacity=1.0,
            specular=0.0,
            specular_power=5,
            ambient=0.2,
            show_edges=False,
        )

    for label in incisor_labels:
        context = pipeline.contexts[label]
        debug = recognizer._ridge_debug(context)
        result = recognizer.recognize(context)
        incisor_debug_data.append((context, debug, result))

        raw_curvature = np.asarray(context.mesh.vertex_mean_curvature[context.vertex_indices], dtype=float)
        clipped_curvature, curvature_clip_min, curvature_clip_max = clip_percentile_range(raw_curvature, 5.0, 95.0)
        curvature_scale = max(abs(float(curvature_clip_min)), abs(float(curvature_clip_max)), 1e-8)
        normalized_curvature = np.clip(clipped_curvature / curvature_scale, -1.0, 1.0)
        full_vertex_curvature[np.asarray(context.vertex_indices, dtype=np.int64)] = normalized_curvature

        ridge_indices = np.asarray(debug["ridge_indices"], dtype=np.int64)
        print(f"Tooth label: {label} ({context.definition.name})")
        print(f"Axis source: {context.axis_source}")
        print(f"Selection source: {debug['selection_source']}")
        print(f"Curvature clip range: [{curvature_clip_min:.4f}, {curvature_clip_max:.4f}]")
        print(f"Ridge point count: {ridge_indices.size}")
        print(f"Midpoint: {result.keypoints[0].point}")

    curvature_plotter = get_dental_plotter()
    pv_mesh_curvature = mesh_to_polydata(mesh)
    pv_mesh_curvature.point_data["ridge_curvature"] = full_vertex_curvature
    curvature_plotter.add_mesh(
        pv_mesh_curvature,
        scalars="ridge_curvature",
        cmap="bwr",
        clim=(-1.0, 1.0),
        opacity=1.0,
        specular=0.0,
        specular_power=5,
        ambient=0.2,
        show_edges=False,
        scalar_bar_args={"title": "Curvature"},
    )
    curvature_plotter.show()

    ridge_plotter = get_dental_plotter()
    add_full_mesh(ridge_plotter)
    for _, debug, result in incisor_debug_data:
        ridge_path_points = np.asarray(debug["ridge_path_points"], dtype=float)
        if ridge_path_points.shape[0] >= 2:
            ridge_plotter.add_mesh(
                pv.lines_from_points(ridge_path_points, close=False),
                color="red",
                line_width=12,
            )
        elif ridge_path_points.shape[0] == 1:
            add_midpoint_sphere(ridge_plotter, ridge_path_points[0], color="red", radius=0.3, opacity=0.8)
        add_midpoint_sphere(ridge_plotter, result.keypoints[0].point, color="green", radius=0.3, opacity=0.8)
    ridge_plotter.show()

    midpoint_plotter = get_dental_plotter()
    add_full_mesh(midpoint_plotter)
    for _, _, result in incisor_debug_data:
        add_midpoint_sphere(midpoint_plotter, result.keypoints[0].point, color="red", radius=0.3, opacity=0.8)
    midpoint_plotter.show()
