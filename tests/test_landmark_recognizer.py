import numpy as np
import trimesh as tm
import pyvista as pv
import matplotlib.pyplot as plt
from pyvista import plotting

from AlveoLab.landmark_recognizer import LandmarkRecognizer
from AlveoLab.pyvista_utils import get_dental_plotter, draw_obb
from AlveoLab.utils import mask_or

def plot_quadratic(lr: LandmarkRecognizer):
    uv = np.c_[lr._mesh.vertices @ lr._orienter.right,
               lr._mesh.vertices @ lr._orienter.forward]

    # draw the curve
    x = np.linspace(np.min(mesh.triangles_center[:, 0]), np.max(mesh.triangles_center[:, 0]), 100)
    y = lr.seg.quadratic.quadratic_2d(x)

    plt.figure(figsize=(6, 6))
    plt.scatter(uv[:, 0], uv[:, 1], s=2, alpha=0.25, label="all vertices")
    plt.plot(x, y, color='red')
    plt.show()


def plot_all_obbs(plotter, lr):
    for group in lr.seg.overlapping_area_groups:
        draw_obb(plotter, group.obb, color=(1, 0.5, 0))


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
    num_peaks = len(lr.peak_indices)

    peak_colors = np.random.rand(num_peaks, 3)

    # assign colors to each face based on the peak region it belongs to
    face_colors = np.ones((lr.mesh.faces.shape[0], 3))
    # for i, group in enumerate(lr._seg.overlapping_area_groups):
    #     mask = group.mask
    #     face_colors[mask] = peak_colors[i]
    for i, tooth in enumerate(lr.seg.teeth):
        mask = tooth.mask
        face_colors[mask] = peak_colors[i]

    # peak_colors = np.random.rand(len(lr._seg.peak_masks), 3)
    # for i, (p, mask) in enumerate(lr._seg.peak_masks.items()):
    #     face_colors[mask] = peak_colors[i]

    return face_colors, peak_colors


if __name__ == '__main__':
    #mesh: tm.Trimesh = tm.load_mesh('../data/1JMandibular_export.stl')
    mesh: tm.Trimesh = tm.load_mesh('../data/models10y/0580_10yr_Maxillary_export.stl')

    landmark_recognizer = LandmarkRecognizer(mesh, 'U')
    mesh = landmark_recognizer.mesh

    # plotting
    # -------------------------
    plotter = get_dental_plotter()
    # plotter.add_bounding_box(color='blue')
    # plotter.add_axes_at_origin(labels_off=True)

    faces_pv = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).flatten()
    pv_mesh = pv.PolyData(mesh.vertices, faces_pv)

    # mesh_color = [1.0, 1.0, 1.0]
    # colors = np.tile(mesh_color, (mesh.faces.shape[0], 1))
    # pv_mesh.cell_data["colors"] = colors
    face_colors, peak_colors = get_spread_regions_color(landmark_recognizer)
    pv_mesh.cell_data["colors"] = face_colors
    # lower, upper = np.percentile(curv, [5, 95])
    hf = landmark_recognizer.harmonic_filed
    lower, upper = np.percentile(hf, [5, 95])
    # 将离群值clamp到这个范围
    hf_clamped = np.clip(hf, lower, upper)
    pv_mesh.point_data["harmonic_filed"] = hf_clamped
    plotter.add_mesh(pv_mesh, scalars="harmonic_filed", opacity=1.0, specular=0.4, specular_power=10, ambient=0.2)

    # plotter.add_mesh(pv_mesh, scalars="colors", rgb=True, opacity=1.0, specular=0.4, specular_power=10, ambient=0.2)

    # add peaks
    # discarded_peaks = np.array(list(landmark_recognizer._seg.discarded_peaks['Spilled Peaks']))
    # discarded_peaks = np.array(list(landmark_recognizer._seg.discarded_overlap_groups["Only on One Side"].peaks))
    # points = mesh.vertices[discarded_peaks]
    # plotter.add_points(points, point_size=20, render_points_as_spheres=True)
    points = mesh.vertices[landmark_recognizer.peak_indices]
    plotter.add_points(points, scalars=peak_colors, rgb=True, point_size=20, render_points_as_spheres=True)

    # plot_all_obbs(plotter, landmark_recognizer)

    plotter.show()

    #plot_horizontal_hull(landmark_recognizer)
    #plot_spread_regions(landmark_recognizer)
    #plot_quadratic(landmark_recognizer)
