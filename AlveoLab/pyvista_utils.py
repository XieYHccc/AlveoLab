import pyvista as pv
from pyvista import plotting
import numpy as np


def get_dental_plotter():
    plot_theme = plotting.themes.DocumentTheme()
    plotter = pv.Plotter(theme=plot_theme, lighting='none')
    plotter.enable_anti_aliasing()
    # plotter.set_background(color='#d8dcd6')  # light grey

    light = pv.Light(color='white', light_type='camera light', intensity=0.8)
    plotter.add_light(light)

    return plotter


def draw_obb(plotter, obb, color=(1, 0, 0)):
    # 创建一个立方体（中心在原点）
    box = pv.Cube(center=(0, 0, 0),
                  x_length=obb.extents[0],
                  y_length=obb.extents[1],
                  z_length=obb.extents[2])

    from_origin_matrix = np.linalg.inv(obb.to_origin_matrix)
    box.transform(from_origin_matrix)

    # 绘制：只显示边
    plotter.add_mesh(box, color="orange", style="wireframe", line_width=2)


