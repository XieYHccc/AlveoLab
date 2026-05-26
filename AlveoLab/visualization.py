import pyvista as pv
import numpy as np
import matplotlib
from trimesh.bounds import oriented_bounds_2D
import matplotlib.pyplot as plt

from AlveoLab.mesh import Mesh
from AlveoLab.pyvista_utils import get_dental_plotter, draw_obb
from AlveoLab.math.geometry import inner_product
from AlveoLab.utils import load_labels
from AlveoLab.segmentation.isoline_voting import extract_loop_candidates, point_in_poly_2d

matplotlib.use("TkAgg")


class LandmarkRecognizerVisualization:
    MESH_BACKGROUND_COLOR = [1.0, 1.0, 1.0]
    #MESH_BACKGROUND_COLOR = [1.0, 0.6, 0.6]

    def __init__(self, landmark_recognizer):
        self.lr = landmark_recognizer
        self.plotter = get_dental_plotter()
        self.mesh = landmark_recognizer.mesh
        self.orienter = self.lr.orienter
        # T = np.array([
        #     [1, 0, 0, 0],
        #     [0, 1, 0, 25],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ])
        # #
        # self.mesh.apply_transform(T)
        # self.mesh.apply_transform(self.orienter.to_origin_transform_matrix)

        self._build_pv_mesh()

        # horizontal projections
        self.uv = np.c_[inner_product(self.mesh.vertices, self.orienter.right),
        inner_product(self.mesh.vertices, self.orienter.forward)]

        # plot settings
        self.mesh_plot_attribute = ''
        self.mesh_plot_attribute_is_rgb = False
        self.cmap = 'viridis'

        self._uniform_hf_isoloops = None

    def _build_uniform_harmonic_isoloops(self, n_isos=80, min_points=20):
        mesh = self.lr.harmonic_seg.dental_mesh
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int64)
        phi = np.asarray(self.lr.harmonic_seg.harmonic_field, dtype=np.float64).reshape(-1)
        lower, upper = np.percentile(phi, [5, 95])
        # 将离群值clamp到这个范围
        phi = np.clip(phi, lower, upper)

        phi_min, phi_max = float(phi.min()), float(phi.max())
        if phi_max - phi_min < 1e-12:
            raise RuntimeError("Harmonic field is near-constant, cannot sample isolines.")
        phi01 = (phi - phi_min) / (phi_max - phi_min)

        loops = extract_loop_candidates(
            V=V,
            F=F,
            phi=phi01,
            right=self.orienter.right,
            forward=self.orienter.forward,
            lo=0.0,
            hi=1.0,
            n_isos=n_isos,
            stitch_tol=1e-4,
            close_tol=2e-3,
            min_points=min_points,
        )
        return loops

    def plot_uniform_harmonic_isolines(self, n_isos=80, line_width=5):
        """从 harmonic field 的 [0,1] 均匀采样并绘制所有闭合 isoloops（colormap=jet_r）。"""
        self.mesh = self.lr.harmonic_seg.dental_mesh
        self._build_pv_mesh()
        loops = self._build_uniform_harmonic_isoloops(n_isos=n_isos)
        self._uniform_hf_isoloops = loops

        cm = matplotlib.colormaps.get_cmap("jet_r")
        for c in loops:
            color = cm(float(np.clip(c.iso, 0.0, 1.0)))[:3]
            self.plot_polyline(c.poly3d, color=color, line_width=line_width)

    def plot_tooth_isoloops_from_peaks(self, peaks, n_isos=80, min_peak_ratio=0.6, line_width=5):
        """
        给定一颗牙的 peaks（Peak 对象列表或顶点索引列表），
        从均匀采样的 isoloops 中筛选“包住这颗牙” 的 loop 并绘制。
        if self._uniform_hf_isoloops is None:
        """
        self._uniform_hf_isoloops = self._build_uniform_harmonic_isoloops(n_isos=n_isos)

        mesh = self.lr.harmonic_seg.dental_mesh
        V = np.asarray(mesh.vertices, dtype=np.float64)

        peak_idx = []
        for p in peaks:
            if hasattr(p, "index"):
                peak_idx.append(int(p.index))
            else:
                peak_idx.append(int(p))
        peak_idx = np.asarray(peak_idx, dtype=np.int64)
        if peak_idx.size == 0:
            return

        peaks3d = V[peak_idx]
        peaks2d = np.c_[peaks3d @ self.orienter.right, peaks3d @ self.orienter.forward]

        cm = matplotlib.colormaps.get_cmap("jet_r")
        for c in self._uniform_hf_isoloops:
            inside = 0
            for p2d in peaks2d:
                if point_in_poly_2d(p2d, c.poly2d):
                    inside += 1
            cov = inside / max(1, len(peaks2d))
            if cov >= min_peak_ratio:
                color = cm(float(np.clip(c.iso, 0.0, 1.0)))[:3]
                self.plot_polyline(c.poly3d, color=color, line_width=line_width)

        for peak in peaks:
            self.plot_sphere_at_point(peak.point, color='blue', radius=0.3)

    def _build_pv_mesh(self):
        # setup pyvista mesh
        faces_pv = np.hstack([np.full((self.mesh.faces.shape[0], 1), 3), self.mesh.faces]).flatten()
        self.pv_mesh = pv.PolyData(self.mesh.vertices, faces_pv)
        colors = np.tile(self.MESH_BACKGROUND_COLOR, (self.mesh.faces.shape[0], 1))
        self.pv_mesh.cell_data["colors"] = colors

    def plot_polyline(self, pts, color="yellow", line_width=4):
        """pts: (N,3) numpy array"""
        pts = np.asarray(pts, dtype=float)
        if pts.shape[0] < 2:
            return
        n = pts.shape[0]
        lines = np.hstack([[n], np.arange(n)]).astype(np.int64)
        poly = pv.PolyData(pts)
        poly.lines = lines
        self.plotter.add_mesh(poly, color=color, line_width=line_width)

    def plot_harmonic_field(self):
        hf = self.lr.harmonic_field

        print(hf.min(), hf.max())
        lower, upper = np.percentile(hf, [5, 95])
        # 将离群值clamp到这个范围
        hf_clamped = np.clip(hf, lower, upper)
        print(hf_clamped.min(), hf_clamped.max())

        faces_pv = np.hstack([np.full((self.lr.harmonic_seg.dental_mesh.faces.shape[0], 1), 3), self.lr.harmonic_seg.dental_mesh.faces]).flatten()
        self.pv_mesh = pv.PolyData(self.lr.harmonic_seg.dental_mesh.vertices, faces_pv)

        self.pv_mesh.point_data["harmonic_filed"] = hf_clamped

        self.mesh_plot_attribute = 'harmonic_filed'
        self.mesh_plot_attribute_is_rgb = False
        self.cmap = 'jet_r'

        # for r in self.lr.harmonic_seg.tooth_boundaries:
        #     if r.best is None:
        #         print("没边界")
        #         continue
        #     self.plot_polyline(r.best.poly3d, color="green", line_width=8)
        # print(len(self.lr.harmonic_seg.tooth_boundaries))

    def plot_hf_contraints_points(self):
        vertices = self.lr.harmonic_seg.dental_mesh.vertices
        for peak_idx in self.lr.harmonic_seg.odd_teeth_peaks:
            self.plot_sphere_at_point(vertices[peak_idx], color='red', radius=0.3)
        for peak_idx in self.lr.harmonic_seg.even_teeth_peaks:
            self.plot_sphere_at_point(vertices[peak_idx], color='blue', radius=0.3)
        # for idx in self.lr.harmonic_seg.gingiva_vertices_indexes:
        #     self.plot_sphere_at_point(vertices[idx], color='green', radius=0.3)

    def plot_height_threshold_plane(self):
        heights = np.inner(self.mesh.vertices, self.orienter.occlusal)  # (N,)
        max_idx = int(np.argmax(heights))  # 最大值对应的vertex索引

        # 计算水平面上的一个点（这里用原点加上法向量乘以高度阈值）
        # point_on_plane = self.mesh.vertices[max_idx] - self.orienter.occlusal * 10
        point_on_plane = self.orienter.occlusal * self.lr.harmonic_seg.cutting_plane_height

        # 创建一个大平面
        plane_size = 80
        plane = pv.Plane(center=point_on_plane, direction=self.orienter.occlusal, i_size=plane_size, j_size=plane_size)

        # 绘制平面，设置半透明
        self.plotter.add_mesh(plane, (98, 96, 170), opacity=0.5, specular=0.0, specular_power=5, ambient=0.2)

    def plot_sphere_at_point(self, point, color, radius=0.5, opacity=0.8):
        self.plotter.add_mesh(pv.Sphere(radius=radius, center=point), color=color, opacity=opacity)

    def plot_valid_peaks(self, color):
        # for peak in self.lr.seg.valid_peaks:
        for peak in self.lr.watershed_filtered_peaks:
            peak_point = peak.point
            self.plot_sphere_at_point(peak_point, color, 0.3, 0.8)

    def plot_discarded_peaks(self, discarded_type, color='red'):
        if discarded_type == "All":
            # 合并所有 discarded_peaks
            all_peaks = set()
            for peaks in self.lr.discarded_peaks.values():
                all_peaks.update(peaks)
            discarded_peaks = np.array(list(all_peaks))
        else:
            discarded_peaks = np.array(
                list(self.lr.discarded_peaks.get(discarded_type, []))
            )

        for peak in discarded_peaks:
            peak_point = peak.point
            self.plot_sphere_at_point(peak_point, color, 0.5, 0.8)

    def plot_peak_masks(self, peak, color):
        peak_masks = self.lr.seg.peak_masks[peak]
        self.pv_mesh.cell_data["colors"][peak_masks] = color
        self.mesh_plot_attribute = "colors"
        self.mesh_plot_attribute_is_rgb = True

    def plot_all_valid_peak_masks(self):
        for peak in self.lr.seg.valid_peaks:
            color = np.random.rand(3)
            peak_masks = self.lr.seg.peak_masks[peak]
            self.pv_mesh.cell_data["colors"][peak_masks] = color

        self.mesh_plot_attribute = "colors"
        self.mesh_plot_attribute_is_rgb = True

    def plot_all_spilled_peak_masks(self):
        spilled_peaks = np.array(
                list(self.lr.discarded_peaks.get('Spilled', []))
            )
        for peak in spilled_peaks:
            color = np.random.rand(3)
            peak_masks = self.lr.seg.peak_masks[peak]
            self.pv_mesh.cell_data["colors"][peak_masks] = color

        self.mesh_plot_attribute = "colors"
        self.mesh_plot_attribute_is_rgb = True
    def plot_all_overlapping_group_masks(self):
        for group in self.lr.seg.overlapping_area_groups:
            colors = np.random.rand(3)
            self.pv_mesh.cell_data["colors"][group.mask] = colors

        self.mesh_plot_attribute = "colors"
        self.mesh_plot_attribute_is_rgb = True

    def plot_teeth(self):
        # for tooth in self.lr.teeth:
        #     colors = np.random.rand(3)
        #     self.pv_mesh.cell_data["colors"][tooth.mask] = colors
        #     self.mesh_plot_attribute = "colors"
        #     self.mesh_plot_attribute_is_rgb = True
        """
        给每颗牙随机上色，并在牙的中心位置标注编号（例如 palmer 或索引），
        文本颜色和牙的颜色一致。
        """
        # 预先取出三角面中心，后面算每颗牙的质心用
        tri_centers = self.mesh.triangles_center

        for idx, tooth in enumerate(self.lr.teeth):
            # 1. 随机一个颜色
            color = np.random.rand(3)
            # 2. 给这颗牙对应的三角形上色（tooth.mask 是一个 face-level 的布尔掩码或索引）
            self.pv_mesh.cell_data["colors"][tooth.mask] = color
            # 3. 计算这颗牙的大致几何中心（用三角面中心的平均）
            tooth_centers = tri_centers[tooth.mask]
            center = tooth_centers.mean(axis=0)
            # 4. 取牙的“编号”：优先用 palmer，没有就用索引
            if hasattr(tooth, "palmer"):
                label = str(tooth.palmer)
            else:
                label = f"{idx}"
            # # 5. 在牙的中心位置画文字标签，颜色和牙的颜色一致
            # self.plotter.add_point_labels(
            #     [center],          # 一个点
            #     [label],           # 对应的标签
            #     point_size=0,      # 不显示点，只显示文字
            #     font_size=16,
            #     text_color=color,  # 文字颜色 = 牙的颜色
            #     render_points_as_spheres=False,
            #     always_visible=True,
            # )

        self.mesh_plot_attribute = "colors"
        self.mesh_plot_attribute_is_rgb = True

    def plot_harmonic_teeth(self):
        hf = self.lr.harmonic_field
        # self.mesh = self.lr.harmonic_seg.dental_mesh
        # self._build_pv_mesh()

        tooth_masks = self.lr.harmonic_seg.get_original_tooth_region_masks()
        # tooth_masks = np.array(list(tooth_masks))
        print(len(tooth_masks))
        for mask in tooth_masks:
            if mask is None:
                continue
            colors = np.random.rand(3)
            self.pv_mesh.cell_data["colors"][mask] = colors

        self.mesh_plot_attribute = "colors"
        self.mesh_plot_attribute_is_rgb = True

    def plot_discarded_overlapping_areas(self):
        for groups in self.lr.seg.discarded_overlap_groups.values():
            for group in groups:
                colors = np.random.rand(3)
                self.pv_mesh.cell_data["colors"][group.mask] = colors
                self.mesh_plot_attribute = "colors"
                self.mesh_plot_attribute_is_rgb = True

    def plot_orientation_axes(self, scale=6):
        tip_length = 0.25
        tip_radius = 0.1
        shaft_radius = 0.05
        arrow_start = self.orienter.center
        arrow_up = pv.Arrow(arrow_start, self.orienter.up, scale=scale, tip_length=tip_length, tip_radius=tip_radius, shaft_radius=shaft_radius)
        arrow_right = pv.Arrow(arrow_start, self.orienter.right, scale=scale, tip_length=tip_length, tip_radius=tip_radius, shaft_radius=shaft_radius)
        arrow_forward = pv.Arrow(arrow_start, self.orienter.forward, scale=scale, tip_length=tip_length, tip_radius=tip_radius, shaft_radius=shaft_radius)

        self.plotter.add_mesh(arrow_up, color='green')
        self.plotter.add_mesh(arrow_right, color='red')
        self.plotter.add_mesh(arrow_forward, color='blue')
        legend = self.plotter.add_legend([
            ["  Left", "red"],
            ["  Forward", "blue"],
            ["  Up", "green"]
        ], bcolor=None, border=False, face=pv.Arrow(scale=10), loc="upper right", size=(0.1, 0.1))
        legend.GetEntryTextProperty().SetFontSize(3)

    def plot_tooth_orientations(self):
        for tooth in self.lr.teeth:
            center = tooth.odom.centre_of_mass
            arrow_up = pv.Arrow(center, tooth.odom.occlusal.vector, scale=4)
            arrow_right = pv.Arrow(center, tooth.odom.buccal.vector, scale=4)
            arrow_forward = pv.Arrow(center, tooth.odom.distal.vector, scale=4)
            self.plotter.add_mesh(arrow_up, color='green')
            self.plotter.add_mesh(arrow_right, color='red')
            self.plotter.add_mesh(arrow_forward, color='blue')

    def plot_teeth_obbs(self):
        for tooth in self.lr.teeth:
            draw_obb(self.plotter, tooth.obb, color=(1, 0, 0))

    def plot_horizon_components(self):
        plt.scatter(self.uv[:, 0], self.uv[:, 1], s=2, alpha=0.25, label="all vertices")

    def plot_horizon_convex_hull(self):
        # 画所有点 + 凸包折线（闭合）
        order = np.r_[self.lr.horizontal_hull, self.lr.horizontal_hull[0]]  # 闭合回到起点

        plt.scatter(self.uv[:, 0], self.uv[:, 1], s=0.5, alpha=0.9, label="Projected Vertices", color='pink')
        plt.plot(self.uv[order, 0], self.uv[order, 1], "b-", lw=2, label="Convex Hull")

    def plot_horizon_bounding_box(self, ax=None):
        # 1. 原始点（例如你的地平线 hull）
        hull_vertices = self.uv[self.lr.horizontal_hull]

        # 2. 求 OBB 变换和尺寸
        transform, extents = oriented_bounds_2D(hull_vertices)
        w, h = extents

        # 3. 在 OBB 坐标系中构造矩形四个角（中心在原点）
        #    这里用齐次坐标 (x, y, 1) 方便乘 3x3 矩阵
        corners_local = np.array([
            [-w / 2, -h / 2, 1.0],
            [w / 2, -h / 2, 1.0],
            [w / 2, h / 2, 1.0],
            [-w / 2, h / 2, 1.0]
        ])

        # 4. transform 是把 “原始点 -> OBB 坐标系”
        #    所以要画到原始坐标系，需要用它的逆
        T_inv = np.linalg.inv(transform)
        corners_world = (T_inv @ corners_local.T).T[:, :2]

        # 为了闭合矩形，把第一个点再接到最后
        corners_world_closed = np.vstack([corners_world, corners_world[0]])

        # 5. 画图

        # # 画原始点
        # plt.scatter(hull_vertices[:, 0], hull_vertices[:, 1], s=5, alpha=0.4, label="hull points")

        # 画 OBB
        plt.plot(corners_world_closed[:, 0], corners_world_closed[:, 1], linewidth=2, label="Minmum bounding box",
                 color='red')

        # # 所有点
        # plt.scatter(self.uv[:, 0], self.uv[:, 1], s=2, alpha=0.25, label="all vertices")
        plt.legend()

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

        vertex_weights = np.divide(vertex_weights, counts, out=np.zeros_like(vertex_weights), where=counts != 0)
        vertex_weights = np.maximum(-vertex_weights, 0.0)
        self.pv_mesh.point_data["vertex_weights"] = vertex_weights
        self.mesh_plot_attribute = "vertex_weights"
        self.mesh_plot_attribute_is_rgb = False
        self.cmap = 'Reds'

    def plot_dental_quadratic(self):
        x = np.linspace(-30, 30, 600)
        y = self.lr.seg.quadratic.quadratic_2d(x)
        points_2d = np.column_stack((x, y))
        points_3d = self.lr.seg.quadratic.to_3d(points_2d)

        # 沿 up 方向往上推
        up = np.asarray(self.orienter.occlusal, dtype=float)
        up /= (np.linalg.norm(up) + 1e-12)
        points_3d = points_3d - up * 5

        n_points = len(points_3d)
        lines = np.hstack([[n_points], np.arange(n_points)])
        curve = pv.PolyData(points_3d)
        curve.lines = lines

        self.plotter.add_mesh(curve, color='red', line_width=3)

    def plot_mesh(self):
        if self.mesh_plot_attribute_is_rgb:
            self.plotter.add_mesh(self.pv_mesh, scalars=self.mesh_plot_attribute, rgb=True,
                                  opacity=1.0, specular=0.0, specular_power=5, ambient=0.2)
        elif self.mesh_plot_attribute != '':
            self.plotter.add_mesh(self.pv_mesh, scalars=self.mesh_plot_attribute, cmap=self.cmap,
                                  opacity=1.0, specular=0.0, specular_power=5, ambient=0.2)
        else:
            self.plotter.add_mesh(self.pv_mesh, color=self.MESH_BACKGROUND_COLOR,
                                  opacity=1.0, specular=0.0, specular_power=5, ambient=0.2)

        # self.plotter.add_mesh(self.pv_mesh,point_size=3,render_points_as_spheres=True,color="pink")

    def plot_mesh_with_vertex_scalar(self, scalar, name="vertex_scalar", cmap="viridis", clip_percentile=None):
        scalar = np.asarray(scalar).reshape(-1)

        if scalar.shape[0] != self.pv_mesh.n_points:
            raise ValueError(
                f"scalar length ({scalar.shape[0]}) must match number of mesh vertices ({self.pv_mesh.n_points})."
            )

        if self.mesh_plot_attribute_is_rgb:
            self.mesh_plot_attribute_is_rgb = False

        if clip_percentile is not None:
            lower, upper = np.percentile(scalar, clip_percentile)
            scalar = np.clip(scalar, lower, upper)

        self.pv_mesh.point_data[name] = scalar
        self.mesh_plot_attribute = name
        self.cmap = cmap

        self.plotter.add_mesh(
            self.pv_mesh,
            scalars=name,
            cmap=self.cmap,
            opacity=1.0,
            specular=0.0,
            specular_power=5,
            ambient=0.2,
        )



    def show(self):
        self.plotter.show()
        plt.show()

    def draw_world_axes_lines(
            self,
            length=80,  # 总长度（-L/2 到 +L/2）
            line_width=8,  # 线宽（像素）
            opacity=1.0,
            show_origin_label=False,
            label_font_size=14,
            # 新增：箭头参数
            arrow_scale=0.03,  # 箭头相对length的比例
            arrow_tip_length=0.95,  # 箭头尖长度比例（相对箭头本身）
            arrow_tip_radius=0.08,  # 尖半径比例（相对箭头本身）
            arrow_shaft_radius=0.01  # 杆半径比例（相对箭头本身）
    ):
        """
        在原点画默认世界坐标轴 X/Y/Z（轴线 + 末端箭头）。
        轴范围：[-L/2, +L/2]，穿过模型。
        """

        half = float(length) / 2.0
        O = np.array([0.0, 0.0, 0.0], dtype=float)

        dirs = np.array([
            [1.0, 0.0, 0.0],  # X
            [0.0, 1.0, 0.0],  # Y
            [0.0, 0.0, 1.0],  # Z
        ], dtype=float)

        # 固定：X红、Y绿、Z蓝
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        names = ["X", "Y", "Z"]

        # 箭头大小（世界坐标）
        arrow_len = float(length) * float(arrow_scale)

        for d, c, name in zip(dirs, colors, names):
            p0 = O - d * half
            p1 = O + d * half

            # 轴线：p0 -> p1
            line = pv.Line(p0, p1)
            self.plotter.add_mesh(
                line,
                color=c,
                line_width=line_width,
                opacity=opacity,
                render_lines_as_tubes=False,  # 不要tube效果
            )

            # 末端箭头：从 (p1 - d*arrow_len) 指向 p1
            start = p1 - d * arrow_len
            arrow = pv.Arrow(
                start=start,
                direction=d,
                tip_length=arrow_tip_length,
                tip_radius=arrow_len * arrow_tip_radius,
                shaft_radius=arrow_len * arrow_shaft_radius,
                scale=arrow_len,  # 箭头整体长度
            )
            self.plotter.add_mesh(
                arrow,
                color=c,
                opacity=opacity,
                smooth_shading=True,
            )

            # 末端文字
            self.plotter.add_point_labels(
                [p1],
                [name],
                font_size=label_font_size,
                text_color=c,
                point_size=0,
                shape_opacity=0.0,
                always_visible=True,
            )

        if show_origin_label:
            self.plotter.add_point_labels(
                [O],
                ["O"],
                font_size=label_font_size,
                text_color="black",
                point_size=0,
                shape_opacity=0.0,
                always_visible=True,
            )

    # for draw two meshes
    def add_mesh(self, trimesh):
        # setup pyvista mesh
        faces_pv = np.hstack([np.full((trimesh.faces.shape[0], 1), 3), trimesh.faces]).flatten()
        pv_mesh = pv.PolyData(trimesh.vertices, faces_pv)
        colors = np.tile(self.MESH_BACKGROUND_COLOR, (trimesh.faces.shape[0], 1))
        pv_mesh.cell_data["colors"] = colors
        self.plotter.add_mesh(pv_mesh, color=self.MESH_BACKGROUND_COLOR,
                              opacity=1.0, specular=0.2, specular_power=5, ambient=0.2)

    def add_mesh_with_labels(self, mesh, labels):
        palette = np.array([
            [255, 255, 255],  # gingiva
            [153, 76, 0], [153, 153, 0], [76, 153, 0], [0, 153, 153], [0, 0, 153], [153, 0, 153],
            [255, 128, 0], [153, 153, 0], [76, 153, 0], [0, 153, 153], [0, 0, 153], [153, 0, 153],
        ]) / 255

        palette[7:] *= 0.4
        # setup pyvista mesh
        faces_pv = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).flatten()
        pv_mesh = pv.PolyData(mesh.vertices, faces_pv)

        max_id = palette.shape[0] - 1
        # 如果label超出范围：这里用clip兜底；你也可以选择 raise
        labels_safe = np.clip(labels.astype(np.int64), 0, max_id)

        rgb = (palette[labels_safe] * 255).astype(np.uint8)  # (N,3) uint8 更适合pyvista显示
        pv_mesh.point_data["rgb"] = rgb
        self.plotter.add_mesh(pv_mesh, scalars="rgb", rgb=True,
                              opacity=1.0, specular=0.1, specular_power=5, ambient=0.2)

    def plot_peak_accumulative_cost_no_mask(
            self,
            peak,
            cmap="jet_r",  # 低=红，高=蓝（像你示例图）
            clip_percentile=(1, 99),  # 稳定显示，避免极端值拉爆色条
            cap=None,  # 可选固定上限，例如 0.115
            show_scalar_bar=True,
            title="Minimum accumulated curvature costs",
    ):
        """单个 peak 的 accumulative cost（不按 peak_mask 裁剪）"""
        costs = np.asarray(self.lr.seg.peak_costs[peak], dtype=float)  # face-level
        n_faces = self.mesh.faces.shape[0]
        if costs.shape[0] != n_faces:
            raise ValueError("peak_costs 必须是 face-level，长度=mesh.faces.shape[0]")

        finite = np.isfinite(costs)
        if not finite.any():
            return

        # 显示范围
        if cap is not None:
            vmin = float(np.nanmin(costs[finite]))
            vmax = float(cap)
        else:
            vmin, vmax = np.percentile(costs[finite], clip_percentile)
            vmin, vmax = float(vmin), float(vmax)
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6

        # 把 inf / nan 压到 vmax，保证可视化稳定
        scalars = costs.copy()
        scalars[~np.isfinite(scalars)] = vmax
        scalars = np.clip(scalars, vmin, vmax)

        self.pv_mesh.cell_data["acc_cost_no_mask"] = scalars


        self.plotter.add_mesh(
            self.pv_mesh,
            scalars="acc_cost_no_mask",
            cmap=cmap,
            clim=(vmin, vmax),
            specular=0.0,
            ambient=0.2,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={"title": title} if show_scalar_bar else None,
        )

        # peak 点
        self.plot_sphere_at_point(peak.point, color="black", radius=0.3, opacity=1.0)

    def plot_all_peaks_accumulative_cost_no_mask(
            self,
            peaks=None,
            mode="min",  # "min" or "mean"
            cmap="jet_r",
            clip_percentile=(1, 99),
            cap=None,
            show_scalar_bar=True,
            title="Accumulated Curvature Costs",
            show_peaks=True,
    ):
        """
        所有 peaks 合成 cost 图（不使用 peak_mask）
        - mode="min": 每个 face 取所有 peak 的最小 cost（最常用）
        - mode="mean": 每个 face 取平均 cost
        """
        if peaks is None:
            peaks = list(self.lr.seg.peaks)

        n_faces = self.mesh.faces.shape[0]
        all_costs = []

        for p in peaks:
            c = np.asarray(self.lr.seg.peak_costs[p], dtype=float)
            if c.shape[0] != n_faces:
                raise ValueError("peak_costs 必须是 face-level")
            all_costs.append(c)

        C = np.vstack(all_costs)  # (n_peaks, n_faces)

        # 把 inf 当作很大值处理，避免 nanmin 报错
        finite_any = np.isfinite(C).any(axis=0)
        if not finite_any.any():
            return

        if mode == "min":
            # 对每个 face 取最小有限值
            C2 = C.copy()
            C2[~np.isfinite(C2)] = np.inf
            s = np.min(C2, axis=0)
        elif mode == "mean":
            C2 = C.copy()
            C2[~np.isfinite(C2)] = np.nan
            s = np.nanmean(C2, axis=0)
        else:
            raise ValueError("mode 只能是 'min' 或 'mean'")

        finite = np.isfinite(s)
        if cap is not None:
            vmin = float(np.nanmin(s[finite]))
            vmax = float(cap)
        else:
            vmin, vmax = np.percentile(s[finite], clip_percentile)
            vmin, vmax = float(vmin), float(vmax)
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6

        s[~np.isfinite(s)] = vmax
        s = np.clip(s, vmin, vmax)

        self.pv_mesh.cell_data["acc_cost_all_no_mask"] = s
        self.plotter.add_mesh(
            self.pv_mesh,
            scalars="acc_cost_all_no_mask",
            cmap=cmap,
            clim=(vmin, vmax),
            specular=0.0,
            ambient=0.2,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={
                "title": title,
                "vertical": False,  # 横向
                "width": 0.45,  # 变短（默认通常更长）
                "height": 0.06,  # 厚度
                "position_x": 0.27,  # 居中一点
                "position_y": 0.03,  # 靠底部
                "fmt": "%.3f",  # 数字格式
                "n_labels": 5,
            } if show_scalar_bar else None,
        )

        if show_peaks:
            for p in peaks:
                self.plot_sphere_at_point(p.point, color="black", radius=0.45, opacity=1.0)

    def plot_final_cut_loop_green(
            self,
            dbg: dict,
            line_width: float = 6.0,
            opacity: float = 1.0,
            render_lines_as_tubes: bool = True,
    ):
        """
        只画 dbg["zfinal"] 对应的最优裁剪平面截交曲线（假设只有一个 loop），颜色固定绿色。
        dbg: landmark_recognizer.harmonic_seg.debug
        """
        import numpy as np
        import pyvista as pv

        z = float(dbg["z_final"])

        mesh_tm = self.mesh

        n = np.asarray(self.orienter.occlusal, dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)

        sec = mesh_tm.section(plane_origin=n * z, plane_normal=n)
        if sec is None or len(sec.discrete) == 0:
            raise RuntimeError(f"zfinal={z:.6f} 截平面没有得到截交曲线")

        pts = np.asarray(sec.discrete[0], dtype=float)  # 只有一个 loop
        npts = pts.shape[0]
        if npts < 2:
            return

        poly = pv.PolyData(pts)
        poly.lines = np.hstack([[npts], np.arange(npts, dtype=np.int64)])

        self.plotter.add_mesh(
            poly,
            color="green",
            line_width=line_width,
            opacity=opacity,
            render_lines_as_tubes=render_lines_as_tubes,
        )

        return z
    def plot_candidate_loops_colored_by_energy(
            self,
            dbg: dict,
            use_norm: bool = True,  # True: candidate_energy_norm；False: raw
            only_single_loop: bool = True,  # 论文风格：只画单闭环
            choose_longest_if_multi: bool = False,  # only_single_loop=False时，多环取最长
            close_tol: float = 1e-3,
            min_loop_points: int = 20,
            cmap: str = "jet_r",
            line_width: float = 4.0,
            opacity: float = 1.0,
            show_scalar_bar: bool = True,
            scalar_title: str = "Candidate energy",
    ):
        """
        画所有candidate对应的截交闭环曲线，颜色由candidate energy决定。
        """
        import pyvista as pv
        import numpy as np

        # ---------- 读 dbg ----------
        if "candidate_z" not in dbg:
            raise ValueError("dbg 缺少 candidate_z")
        z_cand = np.asarray(dbg["candidate_z"], dtype=float).reshape(-1)

        e_key = "candidate_energy_norm" if use_norm else "candidate_energy_raw"
        if e_key not in dbg:
            raise ValueError(f"dbg 缺少 {e_key}")
        e_cand = np.asarray(dbg[e_key], dtype=float).reshape(-1)

        if z_cand.size == 0 or e_cand.size == 0 or z_cand.size != e_cand.size:
            raise ValueError("candidate_z 与 candidate_energy 长度异常")

        mesh_tm = self.mesh._mesh if hasattr(self.mesh, "_mesh") else self.mesh
        n = np.asarray(self.orienter.occlusal, dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)

        # 用全局候选能量范围统一映射颜色
        finite = np.isfinite(e_cand)
        if not finite.any():
            raise RuntimeError("候选能量全是非有限值")
        emin = float(np.nanmin(e_cand[finite]))
        emax = float(np.nanmax(e_cand[finite]))
        if np.isclose(emin, emax):
            emax = emin + 1e-6

        # ---------- 小工具函数 ----------
        def _extract_loops(z):
            sec = mesh_tm.section(plane_origin=n * float(z), plane_normal=n)
            if sec is None:
                return []
            loops = []
            for pl in sec.discrete:
                pl = np.asarray(pl, dtype=float)
                if pl.ndim == 2 and pl.shape[0] >= 3 and pl.shape[1] == 3:
                    if np.linalg.norm(pl[0] - pl[-1]) <= close_tol and pl.shape[0] >= min_loop_points:
                        loops.append(pl)
            return loops

        def _polyline_length(poly):
            if poly.shape[0] < 2:
                return 0.0
            d = np.diff(poly, axis=0)
            return float(np.linalg.norm(d, axis=1).sum())

        # ---------- 逐候选画曲线 ----------
        drawn_any = False
        picked_idx = int(dbg["picked_idx"]) if ("picked_idx" in dbg and dbg["picked_idx"] is not None) else None
        picked_center = None

        for i, (z, e) in enumerate(zip(z_cand, e_cand)):
            loops = _extract_loops(z)
            if len(loops) == 0:
                continue

            chosen_loops = []
            if only_single_loop:
                if len(loops) == 1:
                    chosen_loops = [loops[0]]
                else:
                    continue
            else:
                if choose_longest_if_multi and len(loops) > 1:
                    chosen_loops = [max(loops, key=_polyline_length)]
                else:
                    chosen_loops = loops

            for lp in chosen_loops:
                pts = np.asarray(lp, dtype=float)
                npts = pts.shape[0]
                if npts < 2:
                    continue

                poly = pv.PolyData(pts)
                lines = np.hstack([[npts], np.arange(npts, dtype=np.int64)])
                poly.lines = lines

                # 每条曲线所有点赋同一个能量
                poly.point_data["energy"] = np.full(npts, float(e), dtype=float)

                self.plotter.add_mesh(
                    poly,
                    scalars="energy",
                    cmap=cmap,
                    clim=(emin, emax),
                    line_width=line_width,
                    opacity=opacity,
                    show_scalar_bar=False,  # 统一最后加一次bar
                    render_lines_as_tubes=True,
                )
                drawn_any = True

                if picked_idx is not None and i == picked_idx:
                    picked_center = pts.mean(axis=0)

        if not drawn_any:
            raise RuntimeError("没有画出任何candidate曲线。可尝试放宽 min_loop_points 或 close_tol。")

        # 单独添加一次标尺（避免重复）
        if show_scalar_bar:
            dummy = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
            dummy.lines = np.array([2, 0, 1])
            dummy.point_data["energy"] = np.array([emin, emax], dtype=float)
            self.plotter.add_mesh(
                dummy,
                scalars="energy",
                cmap=cmap,
                clim=(emin, emax),
                opacity=0.0,
                line_width=0.0,
                show_scalar_bar=True,
                scalar_bar_args={"title": scalar_title},
            )

if __name__ == '__main__':
    import trimesh as tm
    from AlveoLab.landmark_recognizer import LandmarkRecognizer

    mesh1 = Mesh.from_file('data/labeld_5year_betterv_objs/VAL6_UpperJaw_030919.obj')
    mesh2 = Mesh.from_file('data/labeld_5year_betterv_objs/0580_5yr_Maxillary_export.obj')
    # labels2 = load_labels('saved/pred_labels_pt_pca/0580_5yr_Maxillary_export.json', False)
    # labels_gt = load_labels('data/labeld_5year_betterv_objs/1023_5 year_Mandibular_export.json', True)

    landmark_recognizer = LandmarkRecognizer(mesh2, 'L')

    viz = LandmarkRecognizerVisualization(landmark_recognizer)
    # viz.plot_discarded_overlapping_areas()
    # labels = landmark_recognizer.teeth_vertex_labels
    # hf = landmark_recognizer.harmonic_field
    # labels = landmark_recognizer.harmonic_seg.harmonic_vertex_labels
    # viz.add_mesh_with_labels(mesh2, labels_gt)
    viz.plot_valid_peaks("red")
    # viz.plot_discarded_peaks("Spilled", color='red')
    # viz.plot_discarded_peaks("All", color='black')
    # viz.plot_discarded_peaks("Gingiva Peaks", color='green')
    # viz.plot_discarded_peaks("No candidate passed", color='yellow')
    # viz.plot_edge_based_curvature()
    # viz.plot_teeth_obbs()
    # viz.plot_orientation_axes()
    # viz.plot_tooth_orientations()
    # viz.plot_horizon_components()
    # viz.plot_horizon_convex_hull()
    # viz.plot_horizon_bounding_box()
    # viz.plot_dental_quadratic()
    # viz.plot_height_threshold_plane()0
    # viz.plot_all_valid_peak_masks()
    # viz.plot_all_overlapping_group_masks()

    # viz.plot_mesh()
    # viz.plotter.add_legend([
    #         ["  OK", "green"],
    #         ["  Geometry-check rejected", "red"],
    #     ], bcolor=None, border=False, face=pv.Sphere(), loc="upper center", size=(0.1, 0.1))


    # viz.plot_peak_accumulative_cost_no_mask(landmark_recognizer.seg.peaks[19])
    # print(landmark_recognizer.seg.peak_max_costs[landmark_recognizer.seg.peaks[19]])
    # landmark_recognizer.seg.parse_spread(landmark_recognizer.seg.peaks[32], 2)
    # viz.plot_peak_masks(landmark_recognizer.seg.peaks[32], color=(0, 1, 0))
    # for idx, peak in enumerate(landmark_recognizer.seg.peaks):
    #     if peak.spilled:
    #         print(idx)
    # viz.plot_harmonic_field()
    # hf = landmark_recognizer.harmonic_field
    #
    # viz.plot_candidate_loops_colored_by_energy(
    #     dbg=landmark_recognizer.harmonic_seg.debug,
    #     use_norm=True,  # 看归一化能量
    #     only_single_loop=True,  # 论文风格
    #     cmap="jet_r",
    #     line_width=5.0,
    #     scalar_title="curvature variance energy"
    # )
    # viz.plot_height_threshold_plane()
    # viz.plot_final_cut_loop_green(dbg=landmark_recognizer.harmonic_seg.debug)
    # viz.plot_harmonic_teeth()
    # viz.plot_uniform_harmonic_isolines()
    # viz.plot_hf_contraints_points()
    # viz.plot_tooth_isoloops_from_peaks(peaks=landmark_recognizer.harmonic_seg.teeth[4].peaks)
    # landmark_recognizer.seg.parse_spread()
    # viz.plot_all_peaks_accumulative_cost_no_mask()
    viz.plot_teeth()
    # viz.plot_all_spilled_peak_masks()
    viz.plot_mesh()
    # viz.add_mesh(mesh1)
    # viz.plot_mesh_with_vertex_scalar(mesh2.vertex_mean_curvature, "curv", clip_percentile=(5, 95))
    viz.show()
