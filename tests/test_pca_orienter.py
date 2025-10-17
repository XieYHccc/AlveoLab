import numpy as np
import trimesh as tm
import pyvista as pv
import matplotlib.pyplot as plt
from pyvista import plotting

from AlveoLab.orienter.pca_orienter import PcaOrienter


def visualize_adjust_axis_y_to_tips(points, forward, occlusal):
    """
    可视化：bin 里的最大高度 + 拟合直线
    - 参数:
        points   : (N,3) 点云，例如 mesh.triangles_center
        forward  : (3,) 向量，forward 方向
        occlusal : (3,) 向量，occlusal 方向
    """
    ys = np.dot(points, forward)
    heights = np.dot(points, occlusal)

    min_height = heights.min()
    bins = np.arange(ys.min() - 1, ys.max() + 1)
    args = np.digitize(ys, bins)
    args = np.clip(args, 0, len(bins)-1)

    # 每个 bin 的最大高度
    max_heights = np.full_like(bins, min_height, dtype=float)
    np.maximum.at(max_heights, args, heights)

    # 权重
    weights = np.abs(bins - bins[::-1])
    weights = weights.max() - weights
    weights *= (max_heights - max_heights.min())**6

    # 加权拟合直线
    yz_line = np.polynomial.Polynomial.fit(bins, max_heights, 1, w=weights)
    slope = yz_line.deriv()(0)

    # ---- 绘图 ----
    plt.figure(figsize=(8,6))
    sc = plt.scatter(bins, max_heights, c=weights, cmap="viridis", s=60,
                     label="Max heights (colored by weight)")
    x_line = np.linspace(bins.min(), bins.max(), 200)
    plt.plot(x_line, yz_line(x_line), "r-", linewidth=2,
             label=f"Fitted line (slope={slope:.3f})")

    plt.colorbar(sc, label="Weight")
    plt.xlabel("Forward projection (y bins)")
    plt.ylabel("Occlusal height (max per bin)")
    plt.title("Bin max heights and fitted line")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_fitted_dental_arch(pca_orienter):

    poly = pca_orienter.fitted_gum_curve

    # draw the curve
    x = np.linspace(np.min(mesh.triangles_center[:, 0]), np.max(mesh.triangles_center[:, 0]), 100)
    y = poly(x)
    plt.plot(x, y, color='red')

    # draw mesh point cloud
    plt.scatter(mesh.triangles_center[:, 0], -mesh.triangles_center[:, 2], cmap='viridis', marker='.', s=0.3, c=weights)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    mesh: tm.Trimesh = tm.load_mesh('../data/models10y/0611_10yr_Maxillary_export.stl')
    orienter = PcaOrienter(mesh, 'U')
    # mesh.apply_transform(orienter.to_origin_transform_matrix)

    # fit the dental arch
    weights = np.dot(mesh.triangles_center, orienter.occlusal)
    weights -= np.min(weights)  # make sure the weights are positive
    weights = weights ** 5

    # Fit a quadratic curve to the points with a weighted fitting.
    poly = np.polynomial.Polynomial.fit(mesh.triangles_center[:, 0], -mesh.triangles_center[:, 2], 2, w=weights)

    # draw the curve
    x = np.linspace(np.min(mesh.triangles_center[:, 0]), np.max(mesh.triangles_center[:, 0]), 100)
    y = poly(x)
    plt.plot(x, y, color='red')

    # draw mesh point cloud
    plt.scatter(mesh.triangles_center[:, 0], -mesh.triangles_center[:, 2], cmap='viridis', marker='.', s=0.3, c=weights)
    plt.colorbar()
    plt.show()

    # plotting
    # -------------------------
    faces_pv = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).flatten()
    pv_mesh = pv.PolyData(mesh.vertices, faces_pv)

    plot_theme = plotting.themes.DocumentTheme()
    plotter = pv.Plotter(theme=plot_theme, lighting='none')
    plotter.enable_anti_aliasing()
    plotter.set_background(color='#d8dcd6')  # light grey
    # plotter.add_bounding_box(color='red', line_width=5)

    # set up lighting
    light = pv.Light(color='white', light_type='camera light', intensity=0.8)
    plotter.add_light(light)

    mesh_color = [0.9, 0.9, 0.9]
    colors = np.tile(mesh_color, (mesh.faces.shape[0], 1))
    pv_mesh.cell_data["colors"] = colors
    plotter.add_mesh(pv_mesh, scalars='colors', rgb=True, opacity=1.0, specular=0.4, specular_power=10, ambient=0.2)

    # arrow_start = orienter.center + (orienter.up * 7) + (orienter.forward * -5)
    arrow_start = orienter.center

    arrow_up = pv.Arrow(arrow_start, orienter.up, scale=6)
    arrow_right = pv.Arrow(arrow_start, orienter.right, scale=6)
    arrow_forward = pv.Arrow(arrow_start, orienter.forward, scale=6)
    # arrow_labels = ['Up', 'Right', 'Forward']
    # plotter.add_point_labels(
    #     [arrow_start + orienter.up * 4, arrow_start + orienter.right * 4, arrow_start + orienter.forward * 4],
    #     arrow_labels, italic=True, font_size=13, point_color='black', point_size=0,
    #     render_points_as_spheres=True, always_visible=True)
    plotter.add_mesh(arrow_up, color='green')
    plotter.add_mesh(arrow_right, color='red')
    plotter.add_mesh(arrow_forward, color='blue')

    # 添加图例框
    plotter.add_legend([
        ["  Right", "red"],
        ["  Forward", "blue"],
        ["  Up", "green"]
    ], bcolor="gray", face=pv.Arrow(), loc="upper right", size=(0.2, 0.2))

    plotter.show()
    #visualize_adjust_axis_y_to_tips(mesh.triangles_center, orienter.forward, orienter.occlusal)