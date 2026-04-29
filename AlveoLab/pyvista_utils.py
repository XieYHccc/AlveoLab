import numpy as np
import pyvista as pv
from pyvista import plotting


def get_dental_plotter(**plotter_kwargs):
    plot_theme = plotting.themes.DocumentTheme()
    plotter = pv.Plotter(theme=plot_theme, lighting="none", **plotter_kwargs)
    plotter.enable_anti_aliasing()

    light = pv.Light(color="white", light_type="camera light", intensity=0.8)
    plotter.add_light(light)
    return plotter


def draw_obb(plotter, obb, color=(1, 0, 0)):
    box = pv.Cube(
        center=(0, 0, 0),
        x_length=obb.extents[0],
        y_length=obb.extents[1],
        z_length=obb.extents[2],
    )

    from_origin_matrix = np.linalg.inv(obb.to_origin_matrix)
    box.transform(from_origin_matrix)
    plotter.add_mesh(box, color=color, style="wireframe", line_width=2)


def mesh_to_polydata(mesh):
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(vertices, faces_pv)


def trimesh_to_polydata(mesh):
    return mesh_to_polydata(mesh)


def add_direction_frame(
    plotter,
    center,
    axis_specs,
    axis_length: float,
    *,
    opacity: float = 1.0,
    prefix: str = "",
    font_size: int = 12,
):
    center = np.asarray(center, dtype=float)
    label_prefix = f"{prefix} " if prefix else ""

    for name, direction, color in axis_specs:
        direction = np.asarray(direction, dtype=float)
        arrow = pv.Arrow(center, direction, scale=axis_length)
        plotter.add_mesh(arrow, color=color, opacity=opacity)
        plotter.add_point_labels(
            [center + direction * axis_length * 1.15],
            [f"{label_prefix}{name}"],
            point_size=0,
            font_size=font_size,
            text_color=color,
            shape_opacity=0.15,
            always_visible=True,
        )
