import numpy as np
import trimesh as tm
import pyvista as pv
import matplotlib.pyplot as plt
from pyvista import plotting

from AlveoLab.landmark_recognizer import LandmarkRecognizer
from AlveoLab.pyvista_utils import get_dental_plotter

def plot_horizontal_hull(lr : LandmarkRecognizer):
    uv = np.c_[lr._mesh.vertices @ lr._orienter.right,
               lr._mesh.vertices @ lr._orienter.forward]

    # 画所有点 + 凸包折线（闭合）
    order = np.r_[lr._horizontal_hull, lr._horizontal_hull[0]]  # 闭合回到起点

    plt.figure(figsize=(6, 6))
    plt.scatter(uv[:, 0], uv[:, 1], s=2, alpha=0.25, label="all vertices")
    plt.plot(uv[order, 0], uv[order, 1], "r-", lw=2, label="convex hull")
    # 可选：轻微填充显示凸包区域
    # plt.fill(uv[order,0], uv[order,1], alpha=0.05, color="r")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("u = V·right")
    plt.ylabel("v = V·forward")
    plt.title("2D projection & convex hull")
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_spread_regions_color(lr : LandmarkRecognizer):
    # assign random colors to each peak region
    num_peaks = len(lr._peak_indices)
    num_groups = len(lr._group_region_mask)

    peak_colors = np.random.rand(num_peaks, 3)

    # assign colors to each face based on the peak region it belongs to
    face_colors = np.ones((lr._mesh.faces.shape[0], 3))
    for i, (key, val) in enumerate(lr._group_region_mask.items()):
        mask = lr._group_region_mask[key]
        face_colors[mask] = peak_colors[i]

    # for i, (key, val) in enumerate(lr._seg.peak_region_mask.items()):
    #     mask = lr._seg.peak_region_mask[key]
    #     face_colors[mask] = peak_colors[i]


    return face_colors, peak_colors

if __name__ == '__main__':
    #mesh: tm.Trimesh = tm.load_mesh('../data/1JMandibular_export.stl')
    mesh: tm.Trimesh = tm.load_mesh('../data/models10y/0611_10yr_Maxillary_export.stl')

    landmark_recognizer = LandmarkRecognizer(mesh, 'U')

    # plotting
    # -------------------------
    plotter = get_dental_plotter()
    # plotter.add_bounding_box(color='blue')
    # plotter.add_axes_at_origin(labels_off=True)

    faces_pv = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).flatten()
    pv_mesh = pv.PolyData(mesh.vertices, faces_pv)

    mesh_color = [1.0, 1.0, 1.0]
    #colors = np.tile(mesh_color, (mesh.faces.shape[0], 1))
    #pv_mesh.cell_data["colors"] = colors
    face_colors, peak_colors = get_spread_regions_color(landmark_recognizer)
    pv_mesh.cell_data["colors"] = face_colors
    plotter.add_mesh(pv_mesh, scalars="colors", rgb=True, opacity=1.0, specular=0.4, specular_power=10, ambient=0.2)

    # add peaks
    points = mesh.vertices[landmark_recognizer._peak_indices]
    plotter.add_points(points, scalars=peak_colors, rgb=True, point_size=20, render_points_as_spheres=True)
    plotter.show()

    #plot_horizontal_hull(landmark_recognizer)
    #plot_spread_regions(landmark_recognizer)
