from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from AlveoLab.mhb.models import ToothContext
from AlveoLab.mhb.utils import clip_percentile_range, scale_to_unit_interval


@dataclass
class BasinSummary:
    minimum_local_index: int
    vertex_local_indices: np.ndarray
    size: int
    depth: float
    minimum_height: float
    boundary_local_indices: np.ndarray
    neighbor_minimum_local_indices: np.ndarray
    peak_elevation: float = float("nan")
    upper_quantile_elevation: float = float("nan")


@dataclass
class WatershedResult:
    height_function: np.ndarray
    flow_next_local_indices: np.ndarray
    basin_labels: np.ndarray
    basin_minima_local_indices: np.ndarray
    basin_summaries: dict[int, BasinSummary]


@dataclass
class BasinMergeEvent:
    source_minimum_local_index: int
    target_minimum_local_index: int
    trigger: str
    source_depth: float
    source_size: int


@dataclass
class CuspCandidate:
    local_index: int
    score: float
    basin_size: int
    basin_depth: float
    height_value: float
    curvature_value: float
    elevation_value: float
    peak_elevation_value: float


@dataclass
class CuspDetectionResult:
    candidates: list[CuspCandidate]
    debug: dict[str, Any]


def build_local_vertex_adjacency(context: ToothContext) -> list[np.ndarray]:
    local_lookup = {
        int(vertex_index): local_index
        for local_index, vertex_index in enumerate(np.asarray(context.vertex_indices, dtype=np.int64))
    }

    adjacency: list[np.ndarray] = []
    for vertex_index in np.asarray(context.vertex_indices, dtype=np.int64):
        neighbor_local_indices = sorted(
            {
                local_lookup[int(neighbor)]
                for neighbor in context.mesh.vertex_neighbors[int(vertex_index)]
                if int(neighbor) in local_lookup
            }
        )
        adjacency.append(np.asarray(neighbor_local_indices, dtype=np.int64))

    return adjacency


def compute_vertex_curvature(mesh, vertex_indices: np.ndarray | None = None) -> np.ndarray:
    curvature = np.asarray(mesh.vertex_mean_curvature, dtype=float)
    if vertex_indices is None:
        return curvature
    return np.asarray(curvature[np.asarray(vertex_indices, dtype=np.int64)], dtype=float)


def smooth_scalar_on_adjacency(
    values: np.ndarray,
    adjacency: list[np.ndarray],
    *,
    iterations: int = 0,
) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    if iterations <= 0:
        return values

    for _ in range(int(iterations)):
        smoothed_values = values.copy()
        for local_index, neighbor_local_indices in enumerate(adjacency):
            if neighbor_local_indices.size == 0:
                continue
            sample_indices = np.concatenate((neighbor_local_indices, np.array([local_index], dtype=np.int64)))
            smoothed_values[local_index] = float(np.mean(values[sample_indices]))
        values = smoothed_values
    return values


def normalize_elevation_for_minima(
    elevation: np.ndarray,
    *,
    elevation_sign_for_minima: float = -1.0,
) -> np.ndarray:
    return float(elevation_sign_for_minima) * np.asarray(elevation, dtype=float)


def compute_height_function(
    curvature: np.ndarray | None,
    elevation: np.ndarray,
    *,
    alpha: float = 0.5,
    elevation_sign_for_minima: float = -1.0,
    mode: str = "height_only",
) -> np.ndarray:
    """
    Compute the watershed height function used for cusp-basin extraction.

    `height_only` uses only the sign-adjusted occlusal elevation and ignores `alpha`:
        H(v) = z_min(v)

    `combined` matches Kumar et al. (CAD&A 2012, Sec. 3.2):
        H(v) = -(1 - alpha) * K(v) + alpha * z(v)

    In this codebase, `elevation` is measured along the oriented occlusal axis and
    therefore points toward cusp tips for both arches. To turn cusp extraction into
    a uniform local-minima problem, we first negate the oriented elevation with the
    `elevation_sign_for_minima` step. This plays the same role as the paper's
    lower-arch coordinate negation, but is expressed in the project's oriented frame.
    """
    if mode not in {"height_only", "combined"}:
        raise ValueError("mode must be either 'height_only' or 'combined'.")

    if mode == "combined" and not 0.4 <= float(alpha) <= 0.6:
        raise ValueError("alpha must lie in the interval [0.4, 0.6].")

    normalized_elevation = normalize_elevation_for_minima(
        elevation,
        elevation_sign_for_minima=elevation_sign_for_minima,
    )
    if mode == "height_only":
        return normalized_elevation

    if curvature is None:
        raise ValueError("curvature is required when mode='combined'.")

    curvature = np.asarray(curvature, dtype=float)
    return -(1.0 - float(alpha)) * curvature + float(alpha) * normalized_elevation


def orient_curvature_toward_cusps(
    curvature: np.ndarray,
    elevation: np.ndarray,
    *,
    top_quantile: float = 0.9,
    bottom_quantile: float = 0.5,
) -> tuple[np.ndarray, float]:
    """
    Align curvature sign so that occlusally elevated convex regions tend to have
    positive curvature values before applying the paper's height function.

    The discrete mean-curvature sign depends on mesh normal orientation. If the
    top occlusal region has a lower mean curvature than the rest of the tooth,
    we flip the sign globally for the tooth.
    """
    curvature = np.asarray(curvature, dtype=float)
    elevation = np.asarray(elevation, dtype=float)
    if curvature.size == 0:
        return curvature, 1.0

    top_mask = elevation >= float(np.quantile(elevation, top_quantile))
    bottom_mask = elevation <= float(np.quantile(elevation, bottom_quantile))
    if not np.any(top_mask) or not np.any(bottom_mask):
        return curvature, 1.0

    top_mean = float(np.mean(curvature[top_mask]))
    bottom_mean = float(np.mean(curvature[bottom_mask]))
    if top_mean < bottom_mean:
        return -curvature, -1.0
    return curvature, 1.0


def watershed_basins(
    adjacency: list[np.ndarray],
    height_function: np.ndarray,
    *,
    flat_tolerance: float = 1e-8,
) -> WatershedResult:
    height_function = np.asarray(height_function, dtype=float)
    vertex_count = int(height_function.shape[0])
    if vertex_count != len(adjacency):
        raise ValueError("adjacency length must match the height function length.")

    flow_next = np.full(vertex_count, -1, dtype=np.int64)
    plateau_assigned = np.zeros(vertex_count, dtype=bool)

    for start_local_index in range(vertex_count):
        if plateau_assigned[start_local_index]:
            continue

        target_height = float(height_function[start_local_index])
        plateau_stack = [start_local_index]
        plateau_members: list[int] = []
        plateau_member_set: set[int] = {start_local_index}

        while plateau_stack:
            current_local_index = plateau_stack.pop()
            plateau_members.append(current_local_index)
            for neighbor_local_index in adjacency[current_local_index]:
                neighbor_local_index = int(neighbor_local_index)
                if neighbor_local_index in plateau_member_set:
                    continue
                if abs(float(height_function[neighbor_local_index]) - target_height) > flat_tolerance:
                    continue
                plateau_member_set.add(neighbor_local_index)
                plateau_stack.append(neighbor_local_index)

        plateau_representative = min(plateau_members)
        plateau_parents: dict[int, int] = {plateau_representative: plateau_representative}
        bfs_queue = [plateau_representative]

        while bfs_queue:
            current_local_index = bfs_queue.pop(0)
            for neighbor_local_index in adjacency[current_local_index]:
                neighbor_local_index = int(neighbor_local_index)
                if neighbor_local_index not in plateau_member_set or neighbor_local_index in plateau_parents:
                    continue
                plateau_parents[neighbor_local_index] = current_local_index
                bfs_queue.append(neighbor_local_index)

        lower_neighbors: list[int] = []
        for current_local_index in plateau_members:
            for neighbor_local_index in adjacency[current_local_index]:
                neighbor_local_index = int(neighbor_local_index)
                if neighbor_local_index in plateau_member_set:
                    continue
                if float(height_function[neighbor_local_index]) < target_height - flat_tolerance:
                    lower_neighbors.append(neighbor_local_index)

        if lower_neighbors:
            lower_neighbors = sorted(set(lower_neighbors))
            lower_heights = height_function[lower_neighbors]
            minimum_lower_height = float(lower_heights.min())
            exit_candidates = [
                int(local_index)
                for local_index, neighbor_height in zip(lower_neighbors, lower_heights)
                if abs(float(neighbor_height) - minimum_lower_height) <= flat_tolerance
            ]
            flow_next[plateau_representative] = min(exit_candidates)
        else:
            flow_next[plateau_representative] = plateau_representative

        for current_local_index in plateau_members:
            if current_local_index == plateau_representative:
                continue
            flow_next[current_local_index] = int(plateau_parents[current_local_index])

        plateau_assigned[np.asarray(plateau_members, dtype=np.int64)] = True

    basin_labels = np.full(vertex_count, -1, dtype=np.int64)

    def resolve_minimum(local_index: int) -> int:
        cached = int(basin_labels[local_index])
        if cached >= 0:
            return cached

        next_local_index = int(flow_next[local_index])
        if next_local_index == local_index:
            basin_labels[local_index] = local_index
            return local_index

        minimum_local_index = resolve_minimum(next_local_index)
        basin_labels[local_index] = minimum_local_index
        return minimum_local_index

    for local_index in range(vertex_count):
        resolve_minimum(local_index)

    basin_summaries = summarize_basins(adjacency, height_function, basin_labels)
    basin_minima_local_indices = np.array(sorted(basin_summaries), dtype=np.int64)
    return WatershedResult(
        height_function=height_function,
        flow_next_local_indices=flow_next,
        basin_labels=basin_labels,
        basin_minima_local_indices=basin_minima_local_indices,
        basin_summaries=basin_summaries,
    )


def basin_depth(
    adjacency: list[np.ndarray],
    height_function: np.ndarray,
    basin_labels: np.ndarray,
    basin_minimum_local_index: int,
) -> float:
    basin_minimum_local_index = int(basin_minimum_local_index)
    basin_members = np.flatnonzero(basin_labels == basin_minimum_local_index).astype(np.int64)
    if basin_members.size == 0:
        raise ValueError("Requested basin does not exist in the provided basin labels.")

    boundary_local_indices: list[int] = []
    for local_index in basin_members:
        for neighbor_local_index in adjacency[int(local_index)]:
            if int(basin_labels[int(neighbor_local_index)]) != basin_minimum_local_index:
                boundary_local_indices.append(int(local_index))
                break

    if not boundary_local_indices:
        return float("inf")

    minimum_height = float(height_function[basin_minimum_local_index])
    boundary_heights = height_function[np.asarray(sorted(set(boundary_local_indices)), dtype=np.int64)]
    return float(boundary_heights.min() - minimum_height)


def summarize_basins(
    adjacency: list[np.ndarray],
    height_function: np.ndarray,
    basin_labels: np.ndarray,
    *,
    elevation: np.ndarray | None = None,
) -> dict[int, BasinSummary]:
    basin_labels = np.asarray(basin_labels, dtype=np.int64)
    elevation = None if elevation is None else np.asarray(elevation, dtype=float)
    basin_summaries: dict[int, BasinSummary] = {}

    for basin_minimum_local_index in sorted(np.unique(basin_labels).tolist()):
        basin_minimum_local_index = int(basin_minimum_local_index)
        basin_members = np.flatnonzero(basin_labels == basin_minimum_local_index).astype(np.int64)
        boundary_local_indices: list[int] = []
        neighbor_minima: set[int] = set()

        for local_index in basin_members:
            for neighbor_local_index in adjacency[int(local_index)]:
                neighbor_label = int(basin_labels[int(neighbor_local_index)])
                if neighbor_label == basin_minimum_local_index:
                    continue
                boundary_local_indices.append(int(local_index))
                neighbor_minima.add(neighbor_label)

        basin_summaries[basin_minimum_local_index] = BasinSummary(
            minimum_local_index=basin_minimum_local_index,
            vertex_local_indices=basin_members,
            size=int(basin_members.size),
            depth=basin_depth(adjacency, height_function, basin_labels, basin_minimum_local_index),
            minimum_height=float(height_function[basin_minimum_local_index]),
            boundary_local_indices=np.asarray(sorted(set(boundary_local_indices)), dtype=np.int64),
            neighbor_minimum_local_indices=np.asarray(sorted(neighbor_minima), dtype=np.int64),
            peak_elevation=(
                float(np.max(elevation[basin_members])) if elevation is not None and basin_members.size else float("nan")
            ),
            upper_quantile_elevation=(
                float(np.quantile(elevation[basin_members], 0.9))
                if elevation is not None and basin_members.size
                else float("nan")
            ),
        )

    return basin_summaries


def _merge_target_for_basin(
    adjacency: list[np.ndarray],
    height_function: np.ndarray,
    basin_labels: np.ndarray,
    basin_minimum_local_index: int,
) -> int | None:
    basin_minimum_local_index = int(basin_minimum_local_index)
    basin_members = np.flatnonzero(basin_labels == basin_minimum_local_index).astype(np.int64)
    if basin_members.size == 0:
        return None

    best_key = None
    best_neighbor_basin = None

    for local_index in basin_members:
        for neighbor_local_index in adjacency[int(local_index)]:
            neighbor_local_index = int(neighbor_local_index)
            neighbor_basin = int(basin_labels[neighbor_local_index])
            if neighbor_basin == basin_minimum_local_index:
                continue

            boundary_height = max(float(height_function[local_index]), float(height_function[neighbor_local_index]))
            neighbor_minimum_height = float(height_function[neighbor_basin])
            candidate_key = (boundary_height, neighbor_minimum_height, neighbor_basin)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_neighbor_basin = neighbor_basin

    return best_neighbor_basin


def merge_spurious_basins(
    adjacency: list[np.ndarray],
    height_function: np.ndarray,
    watershed: WatershedResult,
    *,
    min_basin_depth: float = 0.0,
    min_basin_size: int = 0,
) -> tuple[WatershedResult, list[BasinMergeEvent]]:
    basin_labels = np.asarray(watershed.basin_labels, dtype=np.int64).copy()
    merge_events: list[BasinMergeEvent] = []

    while True:
        basin_summaries = summarize_basins(adjacency, height_function, basin_labels)
        if len(basin_summaries) <= 1:
            break

        merge_candidates: list[tuple[float, int, int, str]] = []
        for basin_minimum_local_index, basin_summary in basin_summaries.items():
            if np.isfinite(basin_summary.depth) and basin_summary.depth < float(min_basin_depth):
                merge_candidates.append(
                    (float(basin_summary.depth), basin_summary.size, basin_minimum_local_index, "shallow_depth")
                )
            elif basin_summary.size < int(min_basin_size):
                merge_candidates.append(
                    (float(basin_summary.depth), basin_summary.size, basin_minimum_local_index, "small_size")
                )

        if not merge_candidates:
            break

        merge_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        merged = False
        for _, _, source_basin, trigger in merge_candidates:
            target_basin = _merge_target_for_basin(adjacency, height_function, basin_labels, source_basin)
            if target_basin is None:
                continue

            source_summary = basin_summaries[int(source_basin)]
            basin_labels[basin_labels == int(source_basin)] = int(target_basin)
            merge_events.append(
                BasinMergeEvent(
                    source_minimum_local_index=int(source_basin),
                    target_minimum_local_index=int(target_basin),
                    trigger=trigger,
                    source_depth=float(source_summary.depth),
                    source_size=int(source_summary.size),
                )
            )
            merged = True
            break

        if not merged:
            break

    merged_summaries = summarize_basins(adjacency, height_function, basin_labels)
    merged_minima = np.array(sorted(merged_summaries), dtype=np.int64)
    return (
        WatershedResult(
            height_function=np.asarray(height_function, dtype=float),
            flow_next_local_indices=np.asarray(watershed.flow_next_local_indices, dtype=np.int64),
            basin_labels=basin_labels,
            basin_minima_local_indices=merged_minima,
            basin_summaries=merged_summaries,
        ),
        merge_events,
    )


def extract_cusps_from_basins(
    watershed: WatershedResult,
    curvature: np.ndarray | None,
    elevation: np.ndarray,
    *,
    max_candidates: int | None = None,
    scoring_mode: str = "height_only",
) -> list[CuspCandidate]:
    if scoring_mode not in {"height_only", "combined"}:
        raise ValueError("scoring_mode must be either 'height_only' or 'combined'.")

    curvature = (
        np.full(watershed.height_function.shape, np.nan, dtype=float)
        if curvature is None
        else np.asarray(curvature, dtype=float)
    )
    elevation = np.asarray(elevation, dtype=float)
    basin_summaries = list(watershed.basin_summaries.values())
    if not basin_summaries:
        return []

    raw_depths = np.array(
        [
            basin_summary.depth if np.isfinite(basin_summary.depth) else np.ptp(watershed.height_function)
            for basin_summary in basin_summaries
        ],
        dtype=float,
    )
    basin_depth_scores = scale_to_unit_interval(raw_depths)
    basin_height_scores = scale_to_unit_interval(
        -np.array([basin_summary.minimum_height for basin_summary in basin_summaries], dtype=float)
    )
    basin_size_scores = scale_to_unit_interval(
        np.array([basin_summary.size for basin_summary in basin_summaries], dtype=float)
    )
    basin_peak_elevation_scores = scale_to_unit_interval(
        np.array([basin_summary.peak_elevation for basin_summary in basin_summaries], dtype=float)
    )
    if scoring_mode == "combined":
        basin_curvature_scores = scale_to_unit_interval(
            np.array([curvature[basin_summary.minimum_local_index] for basin_summary in basin_summaries], dtype=float)
        )
    else:
        basin_curvature_scores = np.zeros(len(basin_summaries), dtype=float)

    candidates: list[CuspCandidate] = []
    for summary_index, basin_summary in enumerate(basin_summaries):
        if scoring_mode == "combined":
            score = (
                0.40 * float(basin_depth_scores[summary_index])
                + 0.15 * float(basin_height_scores[summary_index])
                + 0.15 * float(basin_curvature_scores[summary_index])
                + 0.10 * float(basin_size_scores[summary_index])
                + 0.20 * float(basin_peak_elevation_scores[summary_index])
            )
        else:
            score = (
                0.45 * float(basin_depth_scores[summary_index])
                + 0.20 * float(basin_height_scores[summary_index])
                + 0.15 * float(basin_size_scores[summary_index])
                + 0.20 * float(basin_peak_elevation_scores[summary_index])
            )
        candidates.append(
            CuspCandidate(
                local_index=int(basin_summary.minimum_local_index),
                score=score,
                basin_size=int(basin_summary.size),
                basin_depth=float(basin_summary.depth),
                height_value=float(watershed.height_function[basin_summary.minimum_local_index]),
                curvature_value=float(curvature[basin_summary.minimum_local_index]),
                elevation_value=float(elevation[basin_summary.minimum_local_index]),
                peak_elevation_value=float(basin_summary.peak_elevation),
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            -candidate.basin_depth if np.isfinite(candidate.basin_depth) else -np.ptp(watershed.height_function),
            -candidate.basin_size,
            candidate.local_index,
        )
    )
    if max_candidates is not None:
        candidates = candidates[: int(max_candidates)]
    return candidates


def detect_cusps_local_extrema(
    context: ToothContext,
    *,
    max_candidates: int,
    occlusal_quantile: float = 0.8,
    min_separation_ratio: float = 0.18,
    min_separation_mm: float = 0.8,
    curvature_weight: float = 0.3,
    height_weight: float = 0.7,
) -> CuspDetectionResult:
    raw_curvature = np.asarray(compute_vertex_curvature(context.mesh, context.vertex_indices), dtype=float)
    curvature, curvature_sign = orient_curvature_toward_cusps(raw_curvature, context.occlusal_values)
    clipped_curvature, curvature_clip_min, curvature_clip_max = clip_percentile_range(curvature, 5.0, 95.0)
    positive_curvature = np.clip(clipped_curvature, a_min=0.0, a_max=None)
    elevations = np.asarray(context.occlusal_values, dtype=float)
    local_adjacency = build_local_vertex_adjacency(context)

    candidate_indices: list[int] = []
    for local_index, neighbor_local_indices in enumerate(local_adjacency):
        if context.boundary_mask[local_index]:
            continue
        if neighbor_local_indices.size == 0:
            continue
        current_height = float(elevations[local_index])
        neighbor_heights = elevations[neighbor_local_indices]
        if current_height + 1e-6 < float(neighbor_heights.max()):
            continue
        candidate_indices.append(local_index)

    if not candidate_indices:
        fallback_index = int(np.argmax(elevations))
        candidate_indices = [fallback_index]

    candidate_indices_array = np.asarray(candidate_indices, dtype=np.int64)
    candidate_heights = elevations[candidate_indices_array]
    candidate_curvatures = positive_curvature[candidate_indices_array]
    height_cutoff = float(np.quantile(elevations, float(occlusal_quantile)))

    above_cutoff = candidate_heights >= height_cutoff
    if np.any(above_cutoff):
        candidate_indices_array = candidate_indices_array[above_cutoff]
        candidate_heights = candidate_heights[above_cutoff]
        candidate_curvatures = candidate_curvatures[above_cutoff]

    positive_curvature_mask = candidate_curvatures > 0.0
    if np.any(positive_curvature_mask):
        candidate_indices_array = candidate_indices_array[positive_curvature_mask]
        candidate_heights = candidate_heights[positive_curvature_mask]
        candidate_curvatures = candidate_curvatures[positive_curvature_mask]

    if candidate_indices_array.size == 0:
        fallback_index = int(np.argmax(elevations))
        candidate_indices_array = np.array([fallback_index], dtype=np.int64)
        candidate_heights = elevations[candidate_indices_array]
        candidate_curvatures = positive_curvature[candidate_indices_array]

    scores = (
        float(height_weight) * scale_to_unit_interval(candidate_heights)
        + float(curvature_weight) * scale_to_unit_interval(candidate_curvatures)
    )

    min_separation = max(float(min_separation_mm), float(context.horizontal_scale) * float(min_separation_ratio))
    selected_candidates: list[CuspCandidate] = []
    for ordered_index in np.argsort(scores)[::-1]:
        local_index = int(candidate_indices_array[ordered_index])
        if any(
            np.linalg.norm(np.asarray(context.points[local_index], dtype=float) - np.asarray(context.points[selected.local_index], dtype=float))
            < min_separation
            for selected in selected_candidates
        ):
            continue

        selected_candidates.append(
            CuspCandidate(
                local_index=local_index,
                score=float(scores[ordered_index]),
                basin_size=1,
                basin_depth=0.0,
                height_value=float(candidate_heights[ordered_index]),
                curvature_value=float(candidate_curvatures[ordered_index]),
                elevation_value=float(candidate_heights[ordered_index]),
                peak_elevation_value=float(candidate_heights[ordered_index]),
            )
        )
        if len(selected_candidates) >= int(max_candidates):
            break

    return CuspDetectionResult(
        candidates=selected_candidates,
        debug={
            "method": "local_extrema",
            "candidate_count": int(candidate_indices_array.size),
            "raw_local_extrema_local_indices": candidate_indices_array,
            "min_separation": float(min_separation),
            "height_cutoff": float(height_cutoff),
            "curvature_clip_min": float(curvature_clip_min),
            "curvature_clip_max": float(curvature_clip_max),
            "curvature_sign": float(curvature_sign),
        },
    )


def detect_cusps_watershed(
    context: ToothContext,
    *,
    height_function_mode: str = "height_only",
    alpha: float = 0.5,
    max_candidates: int | None = None,
    min_basin_depth: float | None = None,
    min_basin_depth_ratio: float | None = None,
    min_basin_size: int = 3,
    curvature_smoothing_iterations: int = 1,
    curvature_clip_percentiles: tuple[float, float] = (5.0, 95.0),
    top_band_quantile: float = 0.8,
    top_band_relaxation_ratio: float = 0.08,
    flat_tolerance: float = 1e-8,
    include_debug_arrays: bool = False,
) -> CuspDetectionResult:
    elevations = np.asarray(context.occlusal_values, dtype=float)
    local_adjacency = build_local_vertex_adjacency(context)
    raw_curvature: np.ndarray | None = None
    smoothed_curvature: np.ndarray | None = None
    curvature_sign = 1.0
    curvature_clip_min = float("nan")
    curvature_clip_max = float("nan")

    if height_function_mode == "combined":
        raw_curvature = np.asarray(compute_vertex_curvature(context.mesh, context.vertex_indices), dtype=float)
        curvature, curvature_sign = orient_curvature_toward_cusps(raw_curvature, elevations)
        smoothed_curvature = smooth_scalar_on_adjacency(
            curvature,
            local_adjacency,
            iterations=curvature_smoothing_iterations,
        )
        smoothed_curvature, curvature_clip_min, curvature_clip_max = clip_percentile_range(
            smoothed_curvature,
            curvature_clip_percentiles[0],
            curvature_clip_percentiles[1],
        )

    height_function = compute_height_function(
        smoothed_curvature,
        elevations,
        alpha=alpha,
        elevation_sign_for_minima=-1.0,
        mode=height_function_mode,
    )

    initial_watershed = watershed_basins(
        local_adjacency,
        height_function,
        flat_tolerance=flat_tolerance,
    )

    # Height-only basins are driven by a much smoother scalar field than the
    # combined curvature+height field, so they need a gentler default merge
    # threshold to avoid collapsing distinct posterior cusps.
    resolved_min_basin_depth_ratio = (
        float(min_basin_depth_ratio)
        if min_basin_depth_ratio is not None
        else (0.01 if height_function_mode == "height_only" else 0.12)
    )
    resolved_min_basin_depth = (
        float(min_basin_depth)
        if min_basin_depth is not None
        else max(float(np.ptp(height_function)) * resolved_min_basin_depth_ratio, 1e-8)
    )

    merged_watershed, merge_events = merge_spurious_basins(
        local_adjacency,
        height_function,
        initial_watershed,
        min_basin_depth=resolved_min_basin_depth,
        min_basin_size=min_basin_size,
    )

    enriched_summaries = summarize_basins(
        local_adjacency,
        height_function,
        merged_watershed.basin_labels,
        elevation=elevations,
    )
    top_band_threshold = float(np.quantile(elevations, top_band_quantile))
    top_band_margin = float(np.ptp(elevations)) * float(top_band_relaxation_ratio)
    allowed_basin_minima = np.array(
        [
            basin_minimum_local_index
            for basin_minimum_local_index, basin_summary in enriched_summaries.items()
            if basin_summary.peak_elevation >= top_band_threshold - top_band_margin
        ],
        dtype=np.int64,
    )
    if allowed_basin_minima.size == 0:
        allowed_basin_minima = np.asarray(merged_watershed.basin_minima_local_indices, dtype=np.int64)

    gated_summaries = {
        int(basin_minimum_local_index): enriched_summaries[int(basin_minimum_local_index)]
        for basin_minimum_local_index in allowed_basin_minima
    }
    boundary_removed_basin_minima = np.array(
        [
            basin_minimum_local_index
            for basin_minimum_local_index in sorted(gated_summaries)
            if bool(context.boundary_mask[int(basin_minimum_local_index)])
        ],
        dtype=np.int64,
    )
    final_summaries = {
        int(basin_minimum_local_index): basin_summary
        for basin_minimum_local_index, basin_summary in gated_summaries.items()
        if not bool(context.boundary_mask[int(basin_minimum_local_index)])
    }
    if not final_summaries:
        final_summaries = gated_summaries
        boundary_removed_basin_minima = np.empty(0, dtype=np.int64)

    gated_watershed = WatershedResult(
        height_function=np.asarray(merged_watershed.height_function, dtype=float),
        flow_next_local_indices=np.asarray(merged_watershed.flow_next_local_indices, dtype=np.int64),
        basin_labels=np.asarray(merged_watershed.basin_labels, dtype=np.int64),
        basin_minima_local_indices=np.asarray(sorted(final_summaries), dtype=np.int64),
        basin_summaries=final_summaries,
    )

    candidates = extract_cusps_from_basins(
        gated_watershed,
        smoothed_curvature,
        elevations,
        max_candidates=max_candidates,
        scoring_mode=height_function_mode,
    )

    basin_depths = {
        int(basin_minimum_local_index): float(summary.depth)
        for basin_minimum_local_index, summary in final_summaries.items()
    }
    basin_sizes = {
        int(basin_minimum_local_index): int(summary.size)
        for basin_minimum_local_index, summary in final_summaries.items()
    }
    basin_peak_elevations = {
        int(basin_minimum_local_index): float(summary.peak_elevation)
        for basin_minimum_local_index, summary in final_summaries.items()
    }

    debug: dict[str, Any] = {
        "method": "watershed",
        "height_function_mode": height_function_mode,
        "alpha": float(alpha),
        "height_function_formula": (
            "H(v)=z_min(v)"
            if height_function_mode == "height_only"
            else "H(v)=-(1-alpha)*K(v)+alpha*z_min(v)"
        ),
        "height_sign_convention": (
            "Project occlusal elevation points toward cusp tips for both arches; "
            "we negate this oriented elevation to convert cusp extraction into a "
            "uniform local-minima problem, which is equivalent to the paper's "
            "lower-arch coordinate-negation step."
        ),
        "initial_basin_count": int(initial_watershed.basin_minima_local_indices.size),
        "final_basin_count": int(merged_watershed.basin_minima_local_indices.size),
        "removed_by_merging": int(
            initial_watershed.basin_minima_local_indices.size - merged_watershed.basin_minima_local_indices.size
        ),
        "curvature_used_in_height_function": bool(height_function_mode == "combined"),
        "curvature_sign": float(curvature_sign),
        "curvature_smoothing_iterations": (
            int(curvature_smoothing_iterations) if height_function_mode == "combined" else 0
        ),
        "curvature_clip_percentiles": (
            tuple(float(value) for value in curvature_clip_percentiles)
            if height_function_mode == "combined"
            else None
        ),
        "curvature_clip_min": float(curvature_clip_min),
        "curvature_clip_max": float(curvature_clip_max),
        "min_basin_depth": float(resolved_min_basin_depth),
        "min_basin_depth_ratio": float(resolved_min_basin_depth_ratio),
        "min_basin_size": int(min_basin_size),
        "top_band_quantile": float(top_band_quantile),
        "top_band_threshold": float(top_band_threshold),
        "top_band_margin": float(top_band_margin),
        "pre_boundary_filter_basin_minima_local_indices": np.asarray(
            sorted(gated_summaries),
            dtype=np.int64,
        ),
        "boundary_removed_basin_count": int(boundary_removed_basin_minima.size),
        "boundary_removed_basin_minima_local_indices": np.asarray(boundary_removed_basin_minima, dtype=np.int64),
        "basin_depths": basin_depths,
        "basin_sizes": basin_sizes,
        "basin_peak_elevations": basin_peak_elevations,
        "merge_events": [
            {
                "source_minimum_local_index": int(event.source_minimum_local_index),
                "target_minimum_local_index": int(event.target_minimum_local_index),
                "trigger": event.trigger,
                "source_depth": float(event.source_depth),
                "source_size": int(event.source_size),
            }
            for event in merge_events
        ],
        "initial_basin_minima_local_indices": np.asarray(
            initial_watershed.basin_minima_local_indices,
            dtype=np.int64,
        ),
        "merged_basin_minima_local_indices": np.asarray(
            merged_watershed.basin_minima_local_indices,
            dtype=np.int64,
        ),
        "final_basin_minima_local_indices": np.asarray(
            gated_watershed.basin_minima_local_indices,
            dtype=np.int64,
        ),
    }

    if include_debug_arrays:
        debug["raw_height_function"] = np.asarray(height_function, dtype=float)
        debug["initial_basin_labels"] = np.asarray(initial_watershed.basin_labels, dtype=np.int64)
        debug["final_basin_labels"] = np.asarray(gated_watershed.basin_labels, dtype=np.int64)
        if raw_curvature is not None:
            debug["raw_vertex_curvature"] = np.asarray(raw_curvature, dtype=float)
        if smoothed_curvature is not None:
            debug["vertex_curvature"] = np.asarray(smoothed_curvature, dtype=float)
        debug["vertex_elevation"] = np.asarray(elevations, dtype=float)

    return CuspDetectionResult(
        candidates=candidates,
        debug=debug,
    )
