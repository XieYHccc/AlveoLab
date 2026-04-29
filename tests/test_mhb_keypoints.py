import numpy as np
import trimesh as tm
from types import SimpleNamespace

from AlveoLab.mesh import Mesh
from AlveoLab.mhb import (
    INCISOR,
    MOLAR,
    MhbKeypointPipeline,
    TOOTH_LABEL_DEFINITIONS,
)
from AlveoLab.mhb.recognizers.molar import MolarMhbKeypointRecognizer
from AlveoLab.mhb.recognizers.primary_molar import PrimaryMolarMhbKeypointRecognizer


def make_grid_mesh(xs, ys, height_fn, x_offset=0.0):
    vertices = []
    for y in ys:
        for x in xs:
            vertices.append([x + x_offset, y, height_fn(x, y)])
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

    return tm.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)


def attach_incisal_ridge_curvature(mesh: Mesh):
    x = np.asarray(mesh.vertices[:, 0], dtype=float)
    y = np.asarray(mesh.vertices[:, 1], dtype=float)
    curvature = np.full(mesh.vertices.shape[0], -0.25, dtype=float)
    ridge_mask = np.abs(y) < 0.35
    curvature[ridge_mask] = 1.0 - 0.08 * (x[ridge_mask] ** 2)
    mesh._vertex_mean_curvature = curvature
    return mesh


def test_label_mapping_matches_expected_tooth_families():
    assert TOOTH_LABEL_DEFINITIONS[1].family == INCISOR
    assert TOOTH_LABEL_DEFINITIONS[8].family == INCISOR
    assert TOOTH_LABEL_DEFINITIONS[6].family == MOLAR
    assert TOOTH_LABEL_DEFINITIONS[12].family == MOLAR


def test_incisor_recognizer_prefers_ridge_midpoint():
    xs = np.linspace(-3.0, 3.0, 13)
    ys = np.linspace(-1.5, 1.5, 7)

    def ridge_height(x, y):
        return 4.0 - 2.2 * (y ** 2)

    mesh = attach_incisal_ridge_curvature(Mesh(make_grid_mesh(xs, ys, ridge_height)))
    labels = np.ones(mesh.vertices.shape[0], dtype=np.int64)

    pipeline = MhbKeypointPipeline(mesh, labels, occlusal_axis=np.array([0.0, 0.0, 1.0]))
    result = pipeline.results[1]

    assert result.recognizer_name == "incisor"
    assert pipeline.contexts[1].axis_source == "medial_curve"
    assert len(result.keypoints) == 1
    assert result.keypoints[0].kind == "ridge_midpoint"
    assert abs(result.keypoints[0].point[0]) < 0.6
    assert abs(result.keypoints[0].point[1]) < 0.3
    assert result.debug["ridge_point_count"] >= 1


def test_incisor_recognizer_stays_centered_for_tilted_tooth():
    xs = np.linspace(-3.0, 3.0, 13)
    ys = np.linspace(-1.5, 1.5, 7)

    def tilted_ridge_height(x, y):
        return 4.0 - 2.2 * (y ** 2) + 0.9 * x

    mesh = attach_incisal_ridge_curvature(Mesh(make_grid_mesh(xs, ys, tilted_ridge_height)))
    labels = np.ones(mesh.vertices.shape[0], dtype=np.int64)

    pipeline = MhbKeypointPipeline(mesh, labels, occlusal_axis=np.array([0.0, 0.0, 1.0]))
    result = pipeline.results[1]

    assert result.recognizer_name == "incisor"
    assert len(result.keypoints) == 1
    assert abs(result.keypoints[0].point[0]) < 0.6
    assert abs(result.keypoints[0].point[1]) < 0.3
    assert result.debug["ridge_point_count"] >= 1


def test_molar_recognizer_returns_multiple_cusp_candidates():
    xs = np.linspace(-3.0, 3.0, 31)
    ys = np.linspace(-2.0, 2.0, 21)

    def cusp_height(x, y):
        left_peak = 2.7 * np.exp(-((x + 1.4) ** 2 + y ** 2) / 0.25)
        right_peak = 2.6 * np.exp(-((x - 1.4) ** 2 + y ** 2) / 0.25)
        return left_peak + right_peak

    mesh = make_grid_mesh(xs, ys, cusp_height)
    labels = np.full(mesh.vertices.shape[0], 6, dtype=np.int64)

    pipeline = MhbKeypointPipeline(mesh, labels, occlusal_axis=np.array([0.0, 0.0, 1.0]))
    result = pipeline.results[6]

    assert result.recognizer_name == "molar"
    assert len(result.keypoints) >= 2

    cusp_points = np.asarray([kp.point for kp in result.keypoints])
    assert np.any(np.linalg.norm(cusp_points[:, :2] - np.array([-1.4, 0.0]), axis=1) < 0.5)
    assert np.any(np.linalg.norm(cusp_points[:, :2] - np.array([1.4, 0.0]), axis=1) < 0.5)


def test_pipeline_dispatches_recognizers_per_tooth_type():
    xs = np.linspace(-2.0, 2.0, 11)
    ys = np.linspace(-1.2, 1.2, 7)

    incisor_mesh = make_grid_mesh(xs, ys, lambda x, y: 3.5 - 1.8 * (y ** 2), x_offset=-6.0)
    molar_mesh = make_grid_mesh(
        xs,
        ys,
        lambda x, y: 2.0 * np.exp(-((x - 0.7) ** 2 + y ** 2) / 0.2)
        + 2.1 * np.exp(-((x + 0.7) ** 2 + y ** 2) / 0.2),
        x_offset=6.0,
    )

    mesh = tm.util.concatenate([incisor_mesh, molar_mesh])
    labels = np.concatenate(
        [
            np.full(incisor_mesh.vertices.shape[0], 1, dtype=np.int64),
            np.full(molar_mesh.vertices.shape[0], 6, dtype=np.int64),
        ]
    )

    pipeline = MhbKeypointPipeline(mesh, labels, occlusal_axis=np.array([0.0, 0.0, 1.0]))

    assert pipeline.results[1].recognizer_name == "incisor"
    assert pipeline.results[6].recognizer_name == "molar"
    assert pipeline.results[1].tooth_name == "left_central_incisor"
    assert pipeline.results[6].tooth_name == "left_first_molar"


def test_primary_molar_buccal_selection_keeps_only_buccal_cusps():
    recognizer = PrimaryMolarMhbKeypointRecognizer(
        max_buccal_cusps=3,
        buccal_distance_ratio=0.18,
        buccal_distance_mm=1.0,
    )
    context = type(
        "MockToothContext",
        (),
        {"buccolingual_values": np.array([-1.6, -0.2, 0.9, 1.7, 2.3, 2.55], dtype=float)},
    )()

    selected_local_indices, threshold = recognizer._select_buccal_candidate_local_indices(
        context,
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
    )

    assert threshold == 1.0
    np.testing.assert_array_equal(selected_local_indices, np.array([5, 4, 3], dtype=np.int64))


def test_molar_buccal_selection_keeps_only_buccal_cusps():
    recognizer = MolarMhbKeypointRecognizer(
        max_buccal_cusps=3,
        buccal_distance_ratio=0.18,
        buccal_distance_mm=1.0,
    )
    context = type(
        "MockToothContext",
        (),
        {"buccolingual_values": np.array([-1.9, -0.4, 1.0, 1.8, 2.1, 2.7], dtype=float)},
    )()

    selected_local_indices, threshold = recognizer._select_buccal_candidate_local_indices(
        context,
        np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
    )

    assert threshold == 1.0
    np.testing.assert_array_equal(selected_local_indices, np.array([5, 4, 3], dtype=np.int64))


def test_molar_close_cusps_are_merged_by_final_separation_filter():
    recognizer = MolarMhbKeypointRecognizer(
        max_buccal_cusps=4,
        min_cusp_separation_ratio=0.12,
        min_cusp_separation_mm=1.0,
    )
    context = type(
        "MockToothContext",
        (),
        {
            "mesiodistal_values": np.array([10.0, 10.45, 13.0], dtype=float),
            "buccolingual_values": np.array([4.0, 4.1, 3.7], dtype=float),
            "horizontal_scale": 10.0,
        },
    )()
    keypoint_by_local_index = {
        0: SimpleNamespace(score=0.70),
        1: SimpleNamespace(score=0.95),
        2: SimpleNamespace(score=0.80),
    }

    selected_local_indices, threshold, removed_local_indices = recognizer._enforce_min_cusp_separation(
        context,
        np.array([0, 1, 2], dtype=np.int64),
        keypoint_by_local_index,
    )

    assert threshold == 1.2
    np.testing.assert_array_equal(selected_local_indices, np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(removed_local_indices, np.array([0], dtype=np.int64))
