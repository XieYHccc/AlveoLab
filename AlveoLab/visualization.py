import pyvista as pv
import numpy as np
import matplotlib
from trimesh.bounds import oriented_bounds_2D
import matplotlib.pyplot as plt

from AlveoLab.mesh import Mesh
from AlveoLab.pyvista_utils import get_dental_plotter, draw_obb
from AlveoLab.math.geometry import inner_product
from AlveoLab.utils import load_labels

matplotlib.use("TkAgg")


class LandmarkRecognizerVisualization:
    MESH_BACKGROUND_COLOR = [1.0, 1.0, 1.0]
    #MESH_BACKGROUND_COLOR = [1.0, 0.6, 0.6]

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
        self.mesh_plot_attribute = ''
        self.mesh_plot_attribute_is_rgb = False
        self.cmap = 'viridis'

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
        hf = self.lr.harmonic_filed

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

        vertices = self.lr.harmonic_seg.dental_mesh.vertices
        for peak_idx in self.lr.harmonic_seg.odd_teeth_peaks:
            self.plot_sphere_at_point(vertices[peak_idx], color='red')
        for peak_idx in self.lr.harmonic_seg.even_teeth_peaks:
            self.plot_sphere_at_point(vertices[peak_idx], color='blue')
        for idx in self.lr.harmonic_seg.gingiva_vertices_indexes:
            self.plot_sphere_at_point(vertices[idx], color='green')

        # for poly in self.lr.harmonic_seg.best_iso_line.polylines:
        #     self.plot_polyline(poly, color="yellow", line_width=5)
        # self.plot_polyline(self.lr.harmonic_seg.best_iso_line.polylines[2], color="yellow", line_width=5)

        for r in self.lr.harmonic_seg.tooth_boundaries:
            if r.best is None:
                print("没边界")
                continue
            self.plot_polyline(r.best.poly3d, color="white", line_width=5)
        print(len(self.lr.harmonic_seg.tooth_boundaries))

    def plot_height_threshold_plane(self):
        heights = np.inner(self.mesh.vertices, self.orienter.occlusal)  # (N,)
        max_idx = int(np.argmax(heights))  # 最大值对应的vertex索引

        # 计算水平面上的一个点（这里用原点加上法向量乘以高度阈值）
        point_on_plane = self.mesh.vertices[max_idx] - self.orienter.occlusal * 10

        # 创建一个大平面
        plane_size = 200
        plane = pv.Plane(center=point_on_plane, direction=self.orienter.occlusal, i_size=plane_size, j_size=plane_size)

        # 绘制平面，设置半透明
        self.plotter.add_mesh(plane, (98, 96, 170), opacity=0.5, specular=0.0, specular_power=5, ambient=0.2)

    def plot_sphere_at_point(self, point, color, radius=0.5, opacity=0.8):
        self.plotter.add_mesh(pv.Sphere(radius=radius, center=point), color=color, opacity=opacity)

    def plot_valid_peaks(self, color):
        for peak in self.lr.seg.valid_peaks:
            peak_point = peak.point
            self.plot_sphere_at_point(peak_point, color, 0.5, 0.8)

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
        self.plotter.add_legend([
            ["  Right", "red"],
            ["  Forward", "blue"],
            ["  Up", "green"]
        ], bcolor=None, border=False, face=pv.Arrow(), loc="center right", size=(0.1, 0.1))

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

        plt.scatter(self.uv[:, 0], self.uv[:, 1], s=2, alpha=0.9, label="Projected Vertices", color='pink')
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

    def show(self):
        self.plotter.show()
        plt.show()

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
            [255, 153, 153],  # gingiva
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

    def plot_all_candidates_energy_accumulated_heatmap(
            self,
            dbg: dict,
            use_norm: bool = True,  # True: candidate_energy_norm
            close_tol: float = 1e-3,
            min_loop_points: int = 20,
            ring_band_mm: float = 0.8,  # 距离候选平面多近的顶点算“被该候选覆盖”
            weight_mode: str = "inverse",  # "direct" or "inverse"
            aggregate: str = "mean",  # "sum" / "mean" / "max"
            cmap: str = "jet_r",
            clip_percentile=(2, 98),
            show_scalar_bar=True,
            title: str = "All-candidate energy heatmap",
    ):
        """
        把所有candidate能量都投影到mesh上：
        - 对每个candidate z_i，找到距离该平面 <= ring_band_mm 的顶点集合 band_i
        - 将该candidate能量 E_i 累加到 band_i
        - 最后聚合(sum/mean/max)并转成 face-level 画热图
        """

        # ---------- 读取candidate ----------
        if "candidate_z" not in dbg:
            raise ValueError("dbg 缺少 candidate_z")
        z_cand = np.asarray(dbg["candidate_z"], dtype=float).reshape(-1)

        if z_cand.size == 0:
            raise ValueError("candidate_z 为空")

        e_key = "candidate_energy_norm" if use_norm else "candidate_energy_raw"
        if e_key not in dbg:
            raise ValueError(f"dbg 缺少 {e_key}")
        e_cand = np.asarray(dbg[e_key], dtype=float).reshape(-1)
        if e_cand.shape[0] != z_cand.shape[0]:
            raise ValueError("candidate_z 与 energy 长度不一致")

        # ---------- mesh & height ----------
        V = np.asarray(self.mesh.vertices, dtype=float)  # (N,3)
        F = np.asarray(self.mesh.faces, dtype=np.int64)  # (M,3)
        n = np.asarray(self.orienter.occlusal, dtype=float)
        n /= (np.linalg.norm(n) + 1e-12)
        h_v = V @ n  # 每个顶点沿occlusal高度

        N = V.shape[0]

        # 每个候选能量转为“热度权重”
        # direct: 能量越大越热；inverse: 能量越小越热（更像“好平面区域”）
        if weight_mode == "direct":
            w_cand = e_cand.copy()
        elif weight_mode == "inverse":
            # 防止除0：用1-e（norm）或用线性反转
            if use_norm:
                w_cand = 1.0 - np.clip(e_cand, 0.0, 1.0)
            else:
                # raw时做鲁棒归一化再反转
                finite = np.isfinite(e_cand)
                if finite.any():
                    lo, hi = np.percentile(e_cand[finite], [5, 95])
                    if hi - lo > 1e-12:
                        en = np.clip((e_cand - lo) / (hi - lo), 0.0, 1.0)
                    else:
                        en = np.zeros_like(e_cand)
                else:
                    en = np.zeros_like(e_cand)
                w_cand = 1.0 - en
        else:
            raise ValueError("weight_mode 只能是 'direct' 或 'inverse'")

        # ---------- 累加到顶点 ----------
        acc = np.zeros(N, dtype=float)
        cnt = np.zeros(N, dtype=float)
        mx = np.full(N, -np.inf, dtype=float)

        for z, w in zip(z_cand, w_cand):
            band_mask = np.abs(h_v - float(z)) <= float(ring_band_mm)  # 该候选平面带状邻域
            if not np.any(band_mask):
                continue

            if aggregate in ("sum", "mean"):
                acc[band_mask] += float(w)
                cnt[band_mask] += 1.0
            elif aggregate == "max":
                mx[band_mask] = np.maximum(mx[band_mask], float(w))
            else:
                raise ValueError("aggregate 只能是 'sum'/'mean'/'max'")

        if aggregate == "sum":
            vertex_heat = acc
        elif aggregate == "mean":
            vertex_heat = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
        else:  # max
            vertex_heat = np.where(np.isfinite(mx), mx, 0.0)

        # ---------- vertex -> face ----------
        face_heat = vertex_heat[F].mean(axis=1)

        finite = np.isfinite(face_heat)
        if not finite.any():
            raise RuntimeError("face_heat 全是非有限值，检查candidate与ring_band_mm")

        vmin, vmax = np.percentile(face_heat[finite], clip_percentile)
        vmin, vmax = float(vmin), float(vmax)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6

        face_heat = np.clip(face_heat, vmin, vmax)

        # ---------- 上图 ----------
        key = "all_candidates_energy_heat"
        self.pv_mesh.cell_data[key] = face_heat
        self.plotter.add_mesh(
            self.pv_mesh,
            scalars=key,
            cmap=cmap,
            clim=(vmin, vmax),
            opacity=1.0,
            specular=0.0,
            ambient=0.2,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={"title": title} if show_scalar_bar else None,
        )

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
            draw_picked_as_sphere: bool = True,
            picked_sphere_radius: float = 0.8,
            picked_color: str = "white",
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

        # 高亮 picked 候选（可选）
        if draw_picked_as_sphere and picked_center is not None:
            self.plotter.add_mesh(
                pv.Sphere(radius=picked_sphere_radius, center=picked_center),
                color=picked_color,
                opacity=1.0
            )


if __name__ == '__main__':
    import trimesh as tm
    from AlveoLab.landmark_recognizer import LandmarkRecognizer

    # mesh: tm.Trimesh = tm.load_mesh('../data/1JMandibular_export.stl')
    # mesh1: tm.Trimesh = tm.load_mesh('../data/models5y/0609_5 YR_Mandibular_export.stl')
    # mesh: tm.Trimesh = tm.load_mesh('../data/labeld_5year_betterv_objs/0709_5 YR_Mandibular_export.obj')
    # labels1 = load_labels('../data/labeld_5year_betterv_objs/0580_5yr_Maxillary_export.json')

    # mesh2 = Mesh.from_file('../data/models5y/0709_5 YR_Maxillary_export.stl')
    mesh2 = Mesh.from_file('../data/labeld_5year_betterv_objs/0611_5yr_Maxillary_export.obj')
    # mesh2 = Mesh.from_file('../data/models5y/0800_5 year_Maxillary_export.stl')

    landmark_recognizer = LandmarkRecognizer(mesh2, 'L')
    #
    # for peak in landmark_recognizer.seg.peaks:
    #     viz = LandmarkRecognizerVisualization(landmark_recognizer)
    #     viz.plot_valid_peaks("blue")

    #     viz.plot_discarded_peaks("Spilled", color='red')
    #     color = (1, 0, 0) if peak.spilled else (0, 1, 0)
    #     viz.plot_peak_masks(peak, color=color)
    #     viz.plot_mesh()
    #     viz.show()

    viz = LandmarkRecognizerVisualization(landmark_recognizer)
    # viz.plot_teeth()
    # viz.plot_discarded_overlapping_areas()
    # viz.add_mesh_with_labels(mesh1, labels1)
    # viz.plot_valid_peaks("green")
    # viz.plot_discarded_peaks("Spilled", color='red')
    # viz.plot_discarded_peaks("All", color='black')
    # viz.plot_discarded_peaks("Gingiva Peaks", color='green')
    # viz.plot_discarded_peaks("No candidate passed", color='yellow')
    # viz.plot_edge_based_curvature()
    # viz.plot_teeth_obbs()
    # viz.plot_orientation_axes()
    # viz.plot_tooth_orientations()
    # viz.plot_edge_based_curvature()
    # viz.plot_horizon_components()
    # viz.plot_horizon_convex_hull()
    # viz.plot_horizon_bounding_box()
    # viz.plot_dental_quadratic()
    # viz.plot_height_threshold_plane()
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
    viz.plot_harmonic_field()
    hf = landmark_recognizer.harmonic_filed
    # viz.plot_all_candidates_energy_accumulated_heatmap(
    #     dbg=landmark_recognizer.harmonic_seg.debug,
    #     use_norm=True,
    #     weight_mode="inverse",  # 低能=高热
    #     aggregate="mean",  # 对每个顶点取候选平均贡献
    #     ring_band_mm=0.8,  # 0.6~1.2可调
    #     cmap="jet_r",
    #     title="All candidate planes (low energy = hot)"
    # )
    # viz.plot_candidate_loops_colored_by_energy(
    #     dbg=landmark_recognizer.harmonic_seg.debug,
    #     use_norm=True,  # 看归一化能量
    #     only_single_loop=True,  # 论文风格
    #     cmap="jet_r",
    #     line_width=5.0,
    #     scalar_title="Candidate energy (norm)"
    # )
    viz.plot_mesh()
    # landmark_recognizer.seg.parse_spread()
    # viz.plot_all_peaks_accumulative_cost_no_mask()
    viz.show()

