import numpy as np
import trimesh as tm

from AlveoLab.mhb import MhbKeypointPipeline, ToothRegionPcaOrienter, build_arch_medial_curve
from AlveoLab.mhb.medial_curve import ArchMedialCurve, _select_medial_candidates


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


def test_build_arch_medial_curve_returns_curve_for_labeled_teeth():
    left_tooth = make_grid_mesh(
        np.linspace(-2.5, -0.5, 9),
        np.linspace(-1.0, 1.0, 7),
        lambda x, y: 4.0 - 1.2 * (y ** 2),
    )
    right_tooth = make_grid_mesh(
        np.linspace(0.5, 2.5, 9),
        np.linspace(-1.0, 1.0, 7),
        lambda x, y: 4.0 - 1.2 * (y ** 2),
    )

    mesh = tm.util.concatenate([left_tooth, right_tooth])
    labels = np.concatenate(
        [
            np.full(left_tooth.vertices.shape[0], 1, dtype=np.int64),
            np.full(right_tooth.vertices.shape[0], 7, dtype=np.int64),
        ]
    )

    orienter = ToothRegionPcaOrienter(mesh, labels, "L")
    frame = type(
        "Frame",
        (),
        {
            "right": orienter.right,
            "forward": orienter.forward,
            "occlusal": orienter.occlusal,
            "center": orienter.center,
        },
    )()

    medial_curve = build_arch_medial_curve(mesh, labels, frame)

    assert medial_curve is not None
    assert medial_curve.medial_points_2d.shape[0] == 256
    assert medial_curve.strip_half_width > 0.0
    assert medial_curve.candidate_radius >= medial_curve.strip_half_width


def test_select_medial_candidates_stays_local_to_sample_point():
    points_2d = np.array(
        [
            [-0.20, 0.10],
            [0.00, 0.00],
            [0.15, -0.10],
            [0.05, 4.00],
            [-0.10, 4.20],
        ],
        dtype=float,
    )
    point = np.array([0.0, 0.0], dtype=float)
    tangent = np.array([1.0, 0.0], dtype=float)

    nearby_points = _select_medial_candidates(
        points_2d,
        point,
        tangent,
        strip_half_width=0.25,
        candidate_radius=0.75,
        min_points_per_strip=3,
    )

    assert nearby_points.shape[0] == 3
    assert np.all(np.linalg.norm(nearby_points - point[np.newaxis, :], axis=1) <= 0.75 + 1e-8)
    assert np.max(np.abs(nearby_points[:, 1])) < 1.0


def test_pipeline_context_prefers_medial_curve_axes():
    left_tooth = make_grid_mesh(
        np.linspace(-2.5, -0.5, 9),
        np.linspace(-1.0, 1.0, 7),
        lambda x, y: 4.0 - 1.2 * (y ** 2),
    )
    right_tooth = make_grid_mesh(
        np.linspace(0.5, 2.5, 9),
        np.linspace(-1.0, 1.0, 7),
        lambda x, y: 4.0 - 1.2 * (y ** 2),
    )

    mesh = tm.util.concatenate([left_tooth, right_tooth])
    labels = np.concatenate(
        [
            np.full(left_tooth.vertices.shape[0], 1, dtype=np.int64),
            np.full(right_tooth.vertices.shape[0], 7, dtype=np.int64),
        ]
    )

    pipeline = MhbKeypointPipeline(mesh, labels, arch_type="L")

    assert pipeline.medial_curve is not None
    assert pipeline.contexts[1].axis_source == "medial_curve"
    assert pipeline.contexts[7].axis_source == "medial_curve"


def test_estimate_tooth_axes_uses_local_medial_curve_line_fit():
    frame = type(
        "Frame",
        (),
        {
            "right": np.array([1.0, 0.0, 0.0], dtype=float),
            "forward": np.array([0.0, 1.0, 0.0], dtype=float),
            "occlusal": np.array([0.0, 0.0, 1.0], dtype=float),
            "center": np.zeros(3, dtype=float),
        },
    )()

    xs = np.linspace(-4.0, 4.0, 33)
    ys = 0.18 * (xs ** 2)
    medial_points_2d = np.column_stack((xs, ys))
    medial_curve = ArchMedialCurve(
        frame=frame,
        fitted_curve_points_2d=medial_points_2d,
        medial_points_2d=medial_points_2d,
        strip_half_width=0.5,
        candidate_radius=2.0,
        sample_spacing=float(np.mean(np.diff(xs))),
    )

    tooth_xs = np.linspace(1.6, 2.4, 9)
    tooth_ys = 0.18 * (tooth_xs ** 2)
    tooth_points = np.column_stack(
        (
            tooth_xs,
            tooth_ys + np.array([-0.08, -0.05, -0.02, 0.00, 0.03, 0.02, 0.00, -0.03, -0.05], dtype=float),
            np.zeros_like(tooth_xs),
        )
    )

    line_segment_2d, _ = medial_curve.fit_tooth_mesiodistal_line(tooth_points)
    mesiodistal_axis, buccolingual_axis = medial_curve.estimate_tooth_axes(tooth_points)

    expected_tangent_2d = np.array([1.0, 0.72], dtype=float)
    expected_tangent_2d /= np.linalg.norm(expected_tangent_2d)
    fitted_tangent_2d = line_segment_2d[1] - line_segment_2d[0]
    fitted_tangent_2d /= np.linalg.norm(fitted_tangent_2d)

    assert abs(np.dot(fitted_tangent_2d, expected_tangent_2d)) > 0.95
    assert abs(np.dot(mesiodistal_axis[:2], expected_tangent_2d)) > 0.95
    assert abs(np.dot(mesiodistal_axis, buccolingual_axis)) < 1e-6
