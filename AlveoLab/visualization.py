import pyvista as pv
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from AlveoLab.pyvista_utils import get_dental_plotter, draw_obb
from AlveoLab.math.geometry import inner_product

matplotlib.use("TkAgg")


class LandmarkRecognizerVisualization:
    MESH_BACKGROUND_COLOR = [1.0, 1.0, 1.0]

    def __init__(self, landmark_recognizer):
        self.lr = landmark_recognizer
        self.plotter = get_dental_plotter()
        self.mesh = landmark_recognizer.mesh
        self.orienter = self.lr.orienter

        # setup pyvista mesh
        faces_pv = np.hstack([np.full((self.mesh.faces.shape[0], 1), 3), self.mesh.faces]).flatten()
        self.pv_mesh = pv.PolyData(self.mesh.vertices, faces_pv)
        colors = np.tile(self.MESH_BACKGROUND_COLOR, (self.mesh.faces.shape[0], 1))
        self.pv_mesh.cell_data["colors"] = colors

        # horizontal projections
        self.uv = np.c_[inner_product(self.mesh.vertices, self.orienter.right),
                        inner_product(self.mesh.vertices, self.orienter.forward)]

        # plot settings
        self.mesh_plot_attribute =''
        self.mesh_plot_attribute_is_rgb = False
        self.cmap = ''

    def plot_valid_peaks(self):
        for peak_idx in self.lr.peak_indices:
            peak_point = self.lr.mesh.vertices[peak_idx]
            self.plotter.add_mesh(pv.Sphere(radius=0.5, center=peak_point), color='blue')

    def plot_discarded_peaks(self, discarded_type, color='red'):
        discarded_peaks = np.array(list(self.lr.discarded_peaks[discarded_type]))
        for peak_idx in discarded_peaks:
            peak_point = self.lr.mesh.vertices[peak_idx]
            self.plotter.add_mesh(pv.Sphere(radius=0.5, center=peak_point), color=color)

    def plot_teeth(self):
        for tooth in self.lr.teeth:
            colors = np.random.rand(3)
            self.pv_mesh.cell_data["colors"][tooth.mask] = colors
            self.mesh_plot_attribute = "colors"
            self.mesh_plot_attribute_is_rgb = True

    def plot_orientation_axes(self):
        arrow_start = self.orienter.center
        arrow_up = pv.Arrow(arrow_start, self.orienter.up, scale=6)
        arrow_right = pv.Arrow(arrow_start, self.orienter.right, scale=6)
        arrow_forward = pv.Arrow(arrow_start, self.orienter.forward, scale=6)
        # arrow_labels = ['Up', 'Right', 'Forward']
        # self.plotter.add_point_labels(
        #     [arrow_start + self.orienter.up * 4, arrow_start + self.orienter.right * 4, arrow_start + self.orienter.forward * 4],
        #     arrow_labels, italic=True, font_size=13, point_color='black', point_size=0,
        #     render_points_as_spheres=True, always_visible=True)
        self.plotter.add_mesh(arrow_up, color='green')
        self.plotter.add_mesh(arrow_right, color='red')
        self.plotter.add_mesh(arrow_forward, color='blue')
        self.plotter.add_legend([
            ["  Right", "red"],
            ["  Forward", "blue"],
            ["  Up", "green"]
        ], bcolor="gray", face=pv.Arrow(), loc="upper right", size=(0.2, 0.2))

    def plot_teeth_obbs(self):
        for tooth in self.lr.teeth:
            draw_obb(self.plotter, tooth.obb, color=(1, 0, 0))

    def plot_horizon_components(self):
        plt.scatter(self.uv[:, 0], self.uv[:, 1], s=2, alpha=0.25, label="all vertices")

    def plot_horizon_convex_hull(self):
        # 画所有点 + 凸包折线（闭合）
        order = np.r_[self.lr.horizontal_hull, self.lr.horizontal_hull[0]]  # 闭合回到起点

        plt.scatter(self.uv[:, 0], self.uv[:, 1], s=2, alpha=0.25, label="all vertices")
        plt.plot(self.uv[order, 0], self.uv[order, 1], "r-", lw=2, label="convex hull")

    def plot_edge_based_curvature(self):
        edges = self.mesh.face_adjacency_edges

        curvature = self.lr.seg.edge_curvature.copy()
        lower, upper = np.percentile(curvature, [5, 95])
        curvature = np.clip(curvature, lower, upper)
        vertex_weights = np.zeros(self.mesh.vertices.shape[0])
        counts = np.zeros(self.mesh.vertices.shape[0])
        for (i, j), w in zip(edges, curvature):
            vertex_weights[i] += w
            vertex_weights[j] += w
            counts[i] += 1
            counts[j] += 1

        vertex_weights = np.divide(vertex_weights, counts, out=np.zeros_like(vertex_weights), where=counts!=0)
        self.pv_mesh.point_data["vertex_weights"] = vertex_weights
        self.mesh_plot_attribute = "vertex_weights"
        self.mesh_plot_attribute_is_rgb = False
        self.cmap = 'RdBu'

    def plot_dental_quadratic(self):
        x = np.linspace(-30, 30, 400)
        y = self.lr.seg.quadratic.quadratic_2d(x)
        points_2d = np.column_stack((x, y))
        points_3d = self.lr.seg.quadratic.to_3d(points_2d)

        n_points = len(points_3d)
        lines = np.hstack([[n_points], np.arange(n_points)])
        curve = pv.PolyData(points_3d)
        curve.lines = lines

        self.plotter.add_mesh(curve, color='red', line_width=3)

    def show(self):
        if self.mesh_plot_attribute_is_rgb:
            self.plotter.add_mesh(self.pv_mesh, scalars=self.mesh_plot_attribute, rgb=True,
                                  opacity=1.0, specular=0.4, specular_power=10, ambient=0.2)
        else:
            self.plotter.add_mesh(self.pv_mesh, scalars=self.mesh_plot_attribute, cmap=self.cmap,
                                  opacity=1.0, specular=0.4, specular_power=10, ambient=0.2)
        self.plotter.show()

        plt.show()


if __name__ == '__main__':
    import trimesh as tm
    from AlveoLab.landmark_recognizer import LandmarkRecognizer

    # mesh: tm.Trimesh = tm.load_mesh('../data/1JMandibular_export.stl')
    mesh: tm.Trimesh = tm.load_mesh('../data/models10y/0674_10 YR_Mandibular_export.stl')

    landmark_recognizer = LandmarkRecognizer(mesh, 'L')

    viz = LandmarkRecognizerVisualization(landmark_recognizer)
    viz.plot_valid_peaks()
    viz.plot_discarded_peaks("Spilled Peaks", color='red')
    viz.plot_teeth()
    # viz.plot_teeth_obbs()
    viz.plot_orientation_axes()
    #viz.plot_edge_based_curvature()
    #viz.plot_horizon_components()
    #viz.plot_horizon_convex_hull()
    viz.plot_dental_quadratic()
    viz.show()
