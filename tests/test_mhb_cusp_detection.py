import numpy as np
import trimesh as tm

from AlveoLab.mesh import Mesh
from AlveoLab.mhb import MhbKeypointPipeline
from AlveoLab.mhb.cusp_detection import (
    basin_depth,
    compute_height_function,
    detect_cusps_local_extrema,
    detect_cusps_watershed,
    merge_spurious_basins,
    watershed_basins,
)


def make_grid_mesh(xs, ys, height_fn):
    vertices = []
    for y in ys:
        for x in xs:
            vertices.append([x, y, height_fn(x, y)])
    vertices = np.asarray(vertices, dtype=float)

    width = len(xs)
    height = len(ys)
    faces = []
    for row in range(height - 1):
        for col in range(width - 1):
            v0 = row * width + col
            v1 = v0 + 1
            v2 = v0 + width
            v3 = v2 + 1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    return Mesh(tm.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False))


def attach_curvature(mesh: Mesh, curvature: np.ndarray) -> Mesh:
    mesh._vertex_mean_curvature = np.asarray(curvature, dtype=float)
    return mesh


def line_adjacency(length: int) -> list[np.ndarray]:
    adjacency = []
    for index in range(length):
        neighbors = []
        if index > 0:
            neighbors.append(index - 1)
        if index + 1 < length:
            neighbors.append(index + 1)
        adjacency.append(np.asarray(neighbors, dtype=np.int64))
    return adjacency


def test_compute_height_function_uses_uniform_minima_sign_convention_for_combined_mode():
    curvature = np.array([2.0, 4.0], dtype=float)
    elevation = np.array([3.0, 5.0], dtype=float)

    height = compute_height_function(curvature, elevation, alpha=0.5, mode="combined")

    np.testing.assert_allclose(height, np.array([-2.5, -4.5], dtype=float))


def test_compute_height_function_can_use_height_only_mode():
    elevation = np.array([3.0, 5.0], dtype=float)

    height = compute_height_function(None, elevation, mode="height_only")

    np.testing.assert_allclose(height, np.array([-3.0, -5.0], dtype=float))


def test_watershed_basins_flow_to_shared_local_minima():
    heights = np.array([3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0], dtype=float)
    watershed = watershed_basins(line_adjacency(len(heights)), heights)

    assert set(watershed.basin_minima_local_indices.tolist()) == {2, 6}
    assert np.all(watershed.basin_labels[:5] == 2)
    assert np.all(watershed.basin_labels[5:] == 6)


def test_merge_spurious_basins_prefers_shallow_singleton_basin():
    heights = np.array([5.0, 3.0, 1.0, 3.0, 2.7, 3.0, 1.0, 3.0, 5.0], dtype=float)
    adjacency = line_adjacency(len(heights))

    initial = watershed_basins(adjacency, heights)
    assert set(initial.basin_minima_local_indices.tolist()) == {2, 4, 6}
    assert basin_depth(adjacency, heights, initial.basin_labels, 4) == 0.0

    merged, merge_events = merge_spurious_basins(
        adjacency,
        heights,
        initial,
        min_basin_depth=0.05,
        min_basin_size=0,
    )

    assert set(merged.basin_minima_local_indices.tolist()) == {2, 6}
    assert len(merge_events) == 1
    assert merge_events[0].source_minimum_local_index == 4


def test_watershed_cusp_detection_merges_a_shallow_spurious_basin():
    xs = np.linspace(-3.0, 3.0, 31)
    ys = np.linspace(-2.4, 2.4, 25)

    def cusp_height(x, y):
        left_peak = 2.6 * np.exp(-((x + 1.4) ** 2 + y ** 2) / 0.22)
        right_peak = 2.5 * np.exp(-((x - 1.4) ** 2 + y ** 2) / 0.22)
        shallow_bump = 0.75 * np.exp(-((x - 0.1) ** 2 + (y - 0.9) ** 2) / 0.10)
        return left_peak + right_peak + shallow_bump

    mesh = make_grid_mesh(xs, ys, cusp_height)
    attach_curvature(mesh, np.asarray(mesh.vertices[:, 2], dtype=float))
    labels = np.full(mesh.vertices.shape[0], 6, dtype=np.int64)

    pipeline = MhbKeypointPipeline(mesh, labels, occlusal_axis=np.array([0.0, 0.0, 1.0]))
    context = pipeline.contexts[6]

    local_extrema = detect_cusps_local_extrema(context, max_candidates=5, occlusal_quantile=0.75)
    watershed = detect_cusps_watershed(
        context,
        alpha=0.5,
        max_candidates=5,
        min_basin_depth_ratio=0.20,
        min_basin_size=8,
    )

    assert len(local_extrema.candidates) >= 3
    assert len(watershed.candidates) == 2
    assert watershed.debug["removed_by_merging"] >= 1

    cusp_points = np.asarray([context.points[candidate.local_index] for candidate in watershed.candidates], dtype=float)
    assert np.any(np.linalg.norm(cusp_points[:, :2] - np.array([-1.4, 0.0]), axis=1) < 0.5)
    assert np.any(np.linalg.norm(cusp_points[:, :2] - np.array([1.4, 0.0]), axis=1) < 0.5)


def test_watershed_cusp_detection_rejects_boundary_minima():
    xs = np.linspace(-4.0, 4.0, 41)
    ys = np.linspace(-2.4, 2.4, 25)

    def cusp_height(x, y):
        interior_peak = 2.8 * np.exp(-((x + 2.1) ** 2 + y ** 2) / 0.24)
        boundary_peak = 3.0 * np.exp(-(x ** 2 + (y - 0.1) ** 2) / 0.10)
        return interior_peak + boundary_peak

    mesh = make_grid_mesh(xs, ys, cusp_height)
    labels = np.zeros(mesh.vertices.shape[0], dtype=np.int64)
    tooth_mask = np.asarray(mesh.vertices[:, 0], dtype=float) <= 0.0
    labels[tooth_mask] = 6

    pipeline = MhbKeypointPipeline(mesh, labels, occlusal_axis=np.array([0.0, 0.0, 1.0]))
    context = pipeline.contexts[6]
    watershed = detect_cusps_watershed(
        context,
        max_candidates=5,
        min_basin_depth_ratio=0.01,
        min_basin_size=3,
    )

    assert len(watershed.candidates) >= 1
    pre_boundary_filter = np.asarray(watershed.debug["pre_boundary_filter_basin_minima_local_indices"], dtype=np.int64)
    assert any(bool(context.boundary_mask[int(local_index)]) for local_index in pre_boundary_filter)
    assert watershed.debug["boundary_removed_basin_count"] >= 1
    assert all(not bool(context.boundary_mask[candidate.local_index]) for candidate in watershed.candidates)

    cusp_points = np.asarray([context.points[candidate.local_index] for candidate in watershed.candidates], dtype=float)
    assert np.any(np.linalg.norm(cusp_points[:, :2] - np.array([-2.1, 0.0]), axis=1) < 0.6)
