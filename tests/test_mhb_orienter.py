import numpy as np
import trimesh as tm

from AlveoLab.mhb import MhbKeypointPipeline, ToothRegionPcaOrienter
from AlveoLab.orienter.pca_dental_orienter import PcaOrienter


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


def test_tooth_region_orienter_ignores_large_non_tooth_component():
    tooth_mesh = make_grid_mesh(
        np.linspace(-2.0, 2.0, 13),
        np.linspace(-1.0, 1.0, 7),
        lambda x, y: 4.0 - 1.5 * (y ** 2),
        x_offset=0.0,
    )
    gingiva_mesh = make_grid_mesh(
        np.linspace(-6.0, 6.0, 21),
        np.linspace(-4.0, 4.0, 13),
        lambda x, y: -0.2,
        x_offset=30.0,
    )

    mesh = tm.util.concatenate([tooth_mesh, gingiva_mesh])
    labels = np.concatenate(
        [
            np.ones(tooth_mesh.vertices.shape[0], dtype=np.int64),
            np.zeros(gingiva_mesh.vertices.shape[0], dtype=np.int64),
        ]
    )

    full_orienter = PcaOrienter(mesh, "L")
    tooth_orienter = ToothRegionPcaOrienter(mesh, labels, "L")

    assert full_orienter.center[0] > 10.0
    assert abs(tooth_orienter.center[0]) < 2.0
    assert tooth_orienter.selected_vertex_indices.shape[0] == tooth_mesh.vertices.shape[0]
    assert np.dot(tooth_orienter.occlusal, np.array([0.0, 0.0, 1.0])) > 0.5


def test_mhb_pipeline_uses_tooth_region_orientation_by_default():
    tooth_mesh = make_grid_mesh(
        np.linspace(-2.0, 2.0, 13),
        np.linspace(-1.0, 1.0, 7),
        lambda x, y: 4.0 - 1.5 * (y ** 2),
        x_offset=0.0,
    )
    gingiva_mesh = make_grid_mesh(
        np.linspace(-6.0, 6.0, 21),
        np.linspace(-4.0, 4.0, 13),
        lambda x, y: -0.2,
        x_offset=30.0,
    )

    mesh = tm.util.concatenate([tooth_mesh, gingiva_mesh])
    labels = np.concatenate(
        [
            np.ones(tooth_mesh.vertices.shape[0], dtype=np.int64),
            np.zeros(gingiva_mesh.vertices.shape[0], dtype=np.int64),
        ]
    )

    pipeline = MhbKeypointPipeline(mesh, labels, arch_type="L")

    assert isinstance(pipeline.orienter, ToothRegionPcaOrienter)
    assert abs(pipeline.global_frame.center[0]) < 2.0
