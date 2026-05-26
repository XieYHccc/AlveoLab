import numpy as np
import queue
import collections

import AlveoLab.math.geometry as geom
from AlveoLab.math.geometry import inner_product
from AlveoLab.utils import LazyAttribute, mask_or, cached
from AlveoLab.math.least_square_quadratic import Quadratic3D
from AlveoLab.segmentation.overlapping_area_group import OverlappingAreaGroup
from AlveoLab.segmentation.tooth import Tooth
from AlveoLab.grouping import Grouping
from AlveoLab.peak import Peak

def pick_step_threshold(Ts, As, clusters, beta=0.4, min_area_frac=0.35):
    """
    beta: 认为“显著台阶”的阈值，越小越容易保留多个台阶（更倾向选后面的）
    min_area_frac: 选到的阈值下，面积至少达到 As[-1] 的多少（防止太不完整）
    """
    # 1) 给每个 cluster 计算 score
    items = []
    for s, e in clusters:
        if e - s < 2:
            continue
        area_jump = As[e-1] - As[s]
        width = e - s
        score = width * area_jump
        # 候选阈值：台阶开始前一步（更保守）
        Tcand = Ts[max(s-1, 0)]
        Acand = As[max(s-1, 0)]  # 与 Tcand 对齐的累计面积（近似）
        items.append((Tcand, Acand, score, s, e))

    if not items:
        return None, "No clusters"

    scores = np.array([it[2] for it in items], dtype=float)
    max_score = float(scores.max())

    # 2) 取“显著台阶集合”
    strong = [it for it in items if it[2] >= beta * max_score]
    # 3) 在显著台阶里，选 Ts 最大的（更完整）
    strong.sort(key=lambda x: x[0])  # 按 Tcand
    T_pick, A_pick, sc, s, e = strong[-1]

    # 4) 面积不够则再往后挑（如果还有更后面的显著台阶）
    A_final = float(As[-1])
    if A_final > 0 and (A_pick < min_area_frac * A_final) and len(strong) >= 2:
        # 选倒数第二个/或继续往后挑——这里取“更后面那个”已经是最后一个了，
        # 所以面积还不够说明 beta 太高或 min_area_frac 太高，通常调 beta 更小更有效。
        return T_pick, "Picked last strong step but area still small (try lower beta)"

    return T_pick, f"Picked last strong step (beta={beta})"

def cluster_Ts(Ts, tol=1e-3):
    clusters = []
    start = 0
    for i in range(1, len(Ts)):
        if abs(Ts[i] - Ts[start]) > tol:
            clusters.append((start, i))
            start = i
    clusters.append((start, len(Ts)))
    return clusters

def find_stable_dA_then_jump(
    Ts, As,
    W=8,
    lookahead=15,
    # 平台段必须“足够平”：用 dA 的窗口均值来定义
    max_base_dA=5.0,          # 平台段每步平均增量上限（mm^2/step）
    # 跳变判据（在 dA 上）
    jump_ratio=6.0,           # dA 相对平台倍数
    jump_abs_step=30.0        # dA 绝对阈值（mm^2/step）
):
    Ts = np.asarray(Ts, float)
    As = np.asarray(As, float)
    if As.size < W + 3:
        return None, "Too few samples"

    dA = np.diff(As)  # length n-1
    if dA.size < W + 2:
        return None, "Too few dA samples"

    # 1) 找最稳定的一段：窗口均值最小
    win_mean = np.convolve(dA, np.ones(W)/W, mode="valid")  # length dA.size-W+1
    i0 = int(np.argmin(win_mean))
    base = float(win_mean[i0])

    # 平台不够平，直接认为“无平台形态”
    if base > max_base_dA:
        return None, f"No stable plateau in dA (base={base:.2f})"

    # 2) 在平台结束后寻找跳变
    start = i0 + W  # dA 的索引，平台窗口覆盖 [i0 .. i0+W-1]
    end = min(dA.size - 1, start + lookahead)

    for k in range(start, end + 1):
        if (dA[k] > jump_ratio * (base + 1e-9)) and (dA[k] > jump_abs_step):
            # dA[k] 对应 As[k+1]-As[k]，阈值点用 Ts[k]（更保守可用 Ts[k-1]）
            T_pick = Ts[max(k - 1, 0)]
            info = {"base_dA": base, "plateau_dA_range": (i0, i0+W-1), "jump_dA_index": k, "jump_dA": float(dA[k])}
            return T_pick, info

    return None, "Stable plateau found but no jump after it"

def find_plateau_then_jump(
    Ts, As,
    W=6,                 # 平台期最小窗口长度（检查点个数）
    tol_abs=5.0,         # 平台期允许的面积波动（绝对值，mm^2）
    tol_rel=0.03,        # 平台期允许的面积波动（相对值，3%）
    lookahead=10,        # 平台结束后向前看多少个检查点找跳变
    jump_abs=40.0,       # 跳变的绝对面积增量（mm^2）
    jump_ratio=1.6,      # 跳变的倍数（比如 1.6x）
    min_plateau_area=30.0 # 平台面积太小不可信，避免早期误判
):
    """
    返回 (T_jump_before, info_dict) 或 (None, reason)
    """
    Ts = np.asarray(Ts, dtype=float)
    As = np.asarray(As, dtype=float)
    n = As.size
    if n < W + 2:
        return None, "Too few samples"

    # 1) 找所有“平台窗口”：窗口内 As 的波动很小
    stable = np.zeros(n - W + 1, dtype=bool)  # stable[i] 表示窗口 [i, i+W-1] 是否稳定
    for i in range(n - W + 1):
        window = As[i:i+W]
        Amean = window.mean()
        if Amean < min_plateau_area:
            continue
        rng = window.max() - window.min()
        if (rng <= tol_abs) or (rng <= tol_rel * (Amean + 1e-9)):
            stable[i] = True

    if not stable.any():
        return None, "No plateau found"

    # 2) 把连续的 stable 窗口合并成“平台段”
    # stable[i]=True 表示 [i..i+W-1] 稳定，连续窗口会重叠，合并后得到更长平台
    segments = []
    i = 0
    while i < stable.size:
        if not stable[i]:
            i += 1
            continue
        start = i
        while i < stable.size and stable[i]:
            i += 1
        end = i - 1
        # 平台段覆盖的 As 索引是 [start .. end+W-1]
        seg_a = start
        seg_b = end + W - 1
        segments.append((seg_a, seg_b))

    # 3) 对每个平台段，向后 lookahead 找跳变；取“最早发生跳变”的那一个
    best = None
    for (a, b) in segments:
        A_plateau = float(np.mean(As[a:b+1]))
        # 平台结束后的搜索区间
        k0 = b + 1
        k1 = min(n - 1, b + lookahead)
        if k0 >= n:
            continue

        for k in range(k0, k1 + 1):
            if (As[k] - A_plateau >= jump_abs) or (As[k] >= jump_ratio * A_plateau):
                # 跳变发生在 k，阈值取 k-1 更保守
                kk = max(k - 1, 0)
                T_pick = Ts[kk]
                info = {
                    "plateau_range": (a, b),
                    "A_plateau": A_plateau,
                    "jump_at": k,
                    "A_jump": float(As[k]),
                    "T_pick_index": kk
                }
                best = (T_pick, info)
                break
        if best is not None:
            break

    if best is None:
        return None, "Plateau found but no jump after it"

    return best


def nan_maximum(x, y):
    old = np.seterr(invalid="ignore")
    out = np.where((x > y) | np.isnan(y), x, y)
    np.seterr(**old)
    return out

def spillage_threshold(costs, tri2tri_mask):
    """Predict which value of ``max_cost`` will cause a spread to be classed as spilled."""
    return costs[mask_or(*tri2tri_mask.T)].min()

class CurvatureBasedSeg:
    SPREAD_COST_CAP = 5
    MAX_PEAK_DISTANCE = 9
    MAX_TOOTH_WIDTH = 13
    MIN_TOOTH_AREA = 12
    MAX_TOOTH_HEIGHT = 10

    def __init__(self, mesh, orienter, peaks, max_cost="auto"):
        self.mesh = mesh
        self.orienter = orienter
        self.peaks = peaks
        # self._peak_indices = peaks_idx

        self.peak_masks = {}  # peak_id : triangle_mask
        self.peak_costs = {}  # peak_id : accumulative cost to each triangle
        self.teeth = []
        self.overlapping_area_groups = []

        self.discarded_peaks = collections.defaultdict(set)
        self.discarded_overlap_groups = collections.defaultdict(list)  # peaks that are considered as rugae
        self.discarded_teeth = collections.defaultdict(list)
        self.quadratic = None

        self._peak_tri2tri_costs_cache = {}
        self.peak_max_costs = {}
        self.max_cost = max_cost
        self.spillages = []

        self._run()

    @LazyAttribute
    def tri2tri_displacements(self):
        old_settings = np.seterr(divide="ignore", invalid="ignore")
        out = geom.normalize_vector(self.mesh.tri2tri_displacements)
        np.seterr(**old_settings)
        return out

    @LazyAttribute
    def tri2tri_costs(self):
        """The costs, based on curvature, of moving from each triangle to each of it's adjacent
        triangles. It is an array of shape (number of triangles, 3). Triangles are referenced by
        argument based on the order they are listed in the original mesh.
        """
        # We are only looking for the crease where tooth meets gum. Creases / slots / grooves are
        # represented with a negative sign in `mesh.curvature.signed` whereas bumps have positive
        # sign. .clip(max=0) sets all positive values to 0.
        # creases_only = -self.edge_curvature_face_view.clip(max=0)
        creases_only = -self.mesh.tri2tri_edge_curvatures.clip(max=0)

        # An L2 norm seemed to work well, hence the square.
        # return np.ascontiguousarray((creases_only ** 2).clip(max=self.max_cost * 1.1))
        return creases_only.clip(max=self.SPREAD_COST_CAP * 1.1)

    @cached("_peak_tri2tri_costs_cache")
    def peak_tri2tri_costs(self, peak):
        peak_point = peak.point

        in_range_mask = (geom.magnitude_sqr(peak_point - self.mesh.triangles_center) <=
                         self.MAX_PEAK_DISTANCE**2)

        laterals = np.cross(self.mesh.triangles_center[in_range_mask] - peak_point, self.orienter.occlusal)
        lateral_weights = np.abs(
            geom.inner_product(laterals[:, np.newaxis],
                               self.tri2tri_displacements[in_range_mask])) / geom.magnitude(

                                   laterals, keepdims=True)
        #remove nan
        valid_nb = (self.mesh.face_neighbors[in_range_mask] != -1)
        lateral_weights = np.where(valid_nb, lateral_weights, 0.0)
        lateral_weights = np.nan_to_num(lateral_weights, nan=0.0, posinf=0.0, neginf=0.0)

        tri2tri_costs = self.tri2tri_costs[in_range_mask]
        tri_max_costs = geom.reduce_last_ax(tri2tri_costs, nan_maximum, keepdims=True)
        tri2tri_costs = tri_max_costs * lateral_weights + tri2tri_costs * (1 - lateral_weights)
        self.INF_COST = self.SPREAD_COST_CAP * 1.5

        # If no boundary is found (usually if the peak is not on a tooth to begin with) then the
        # recursive spread function will try to fill most of the model, which will take ages.
        # ``mesh.bound_cost_map_by_distance()`` sets all costs that are too far away from the
        # starting point to ``self.INF_COST`` so that the spread function will stop there.
        # `self.MAX_TOOTH_SIZE` is supposed to be the absolute maximum length (in any direction) a
        # single tooth could be.
        #tri2tri_costs = self.mesh.bound_cost_map_by_distance(
        #                    arg, self.MAX_PEAK_DISTANCE, tri2tri_costs, inf_val=self.INF_COST)
        tri2tri_costs_ = np.full_like(self.tri2tri_costs, self.INF_COST)
        tri2tri_costs_[in_range_mask] = tri2tri_costs

        assert not ((self.mesh.face_neighbors != -1) & np.isnan(tri2tri_costs_)).any()

        return tri2tri_costs_

    @LazyAttribute
    def max_cost(self):
        self.max_cost = self.SPREAD_COST_CAP * 2 # 第一次大一点确保accumulative cost 能够蔓延到peak max distance 处
        return self._max_cost

    @max_cost.setter
    def max_cost(self, value="auto"):
        if value == "auto":
            del self.max_cost
        else:
            self._max_cost = value
        # del self.tri2tri_costs
        # self._peak_tri2tri_costs_cache.clear()

    @property
    def discarded_peaks_all(self):
        """Contains all the peak args that we don't want. It comes from flattening
        `self.discarded_peaks`. Any group that contains any of these should be removed. """

        return set().union(*self.discarded_peaks.values())

    @property
    def valid_peaks(self):
        # filtered_peaks = np.array(
        #     [peak for peak in self._peak_indices if peak not in self.discarded_peaks_all])
        # return filtered_peaks

        discarded_idx = {p.index for p in self.discarded_peaks_all}
        return np.array([p for p in self.peaks if p.index not in discarded_idx])


    def _run(self):
        self._build_peaks()
        self._spread_from_peaks()
        self._predict_spillage_thresholds()

        # Step 1: 方案 1 — 全局 coverage maximization 找 T*
        # pick_optimal_max_cost 内部会 try_with_max_cost(T*) 把 quadratic /
        # peak_masks / overlapping_area_groups / teeth 全部建好。
        self.test_max_costs()
        self.pick_optimal_max_cost()
        T_star = float(self.max_cost)

        # Step 2: 方案 C — per-peak 在 [alpha * T_star, T_star] 内收缩。
        # 覆盖 Scheme 1 的 peak_masks(用各自的 T_i),保留 quadratic。
        self._apply_plan_c_thresholds(T_star, alpha=0.6)

        # Step 3: 用新 masks 重建 overlapping_area_groups / teeth。
        # quadratic 可以保持 Scheme 1 阶段的版本(由 valid_peaks 拟合,
        # plan C 的 fallback 行为不会丢 peak,所以集合未变);仍按原顺序重算一遍。
        self._build_overlapping_area_groups()
        self._build_quadratic()
        for group in self.overlapping_area_groups:
            group.update_quadratic(self.quadratic)
        self._group_inline_area_groups()
        self._build_teeth()

    def _build_peaks(self):
        #self.peaks = np.array([Peak(self.mesh.vertices[idx], idx) for idx in self._peak_indices])
        self.peaks = np.array(self.peaks)
        for peak in self.peaks:
            peak.occlusal = self.orienter.occlusal

    def _region_geom_stats(self, mask):
        faces = np.where(mask)[0]
        assert faces.size > 0

        ctr = self.mesh.triangles_center[faces]  # (m,3)
        area = float(self.mesh.area_faces[faces].sum())

        pr = ctr @ self.orienter.right
        pf = ctr @ self.orienter.forward
        po = ctr @ self.orienter.occlusal

        rmin, rmax = float(pr.min()), float(pr.max())
        fmin, fmax = float(pf.min()), float(pf.max())
        omin, omax = float(po.min()), float(po.max())

        width = float(max(rmax - rmin, fmax - fmin))
        height = float(omax - omin)

        # 细长条指标：越大越像“串到邻牙的细带”
        slender = float(width / (np.sqrt(area) + 1e-9))

        return {
            "area": area,
            "width": width,
            "height": height,
            "omin": omin,
            "omax": omax,
            "slender": slender,
            "n_faces": int(faces.size),
        }

    def _two_sided_with_separation(self, faces, side_axis, t=0.35, min_ratio=0.06, sep_ratio=0.25):
        n = self.mesh.face_normals[faces]
        ctr = self.mesh.triangles_center[faces]

        d = n @ side_axis
        pos_mask = d > t
        neg_mask = d < -t

        pos = float(pos_mask.mean())
        neg = float(neg_mask.mean())
        if min(pos, neg) < min_ratio:
            return False

        ps = ctr @ side_axis
        ps_pos = ps[pos_mask]
        ps_neg = ps[neg_mask]

        # 区域在 side_axis 上的跨度
        span = float(ps.max() - ps.min()) + 1e-9
        sep = float(ps_pos.mean() - ps_neg.mean())

        # if sep < sep_ratio * span:
        #     return False, {"reason": "no_spatial_separation", "sep": sep, "span": span, "pos": pos, "neg": neg}

        return True

    def _is_tooth_like_region(self, mask, geo_stats, min_area, max_slender, omin_ref):
        MAX_WIDTH = 11.5
        MAX_HEIGHT = 10.0
        MAX_DROP = 5.0
        MAX_SLENDER = 1.7
        w_width = 0.2
        w_drop = 0.3
        w_slender = 0.2

        faces = np.where(mask)[0]
        area = float(self.mesh.area_faces[faces].sum())
        # if area < min_area:
        #     return False, "too_small"

        ctr = self.mesh.triangles_center[faces]
        pr = ctr @ self.orienter.right
        pf = ctr @ self.orienter.forward

        width = float(max(pr.max() - pr.min(), pf.max() - pf.min()))
        slender = float(width / (np.sqrt(area) + 1e-9))
        if slender > max_slender:
            return False, "too_slender"

        # --- 几何硬过滤 ---
        drop = (omin_ref - geo_stats["omin"])  # drop>0 表示向下走了

        if geo_stats["area"] < min_area or geo_stats["width"] > MAX_WIDTH or geo_stats["height"] > MAX_HEIGHT or \
            geo_stats["slender"] > MAX_SLENDER or drop > MAX_DROP:
            return False, "geometry denyed"


        area = OverlappingAreaGroup(self.peaks[:2], mask, self.mesh, self.orienter, self.quadratic)
        side_axis = area.buccal

        # self._debug_plot_region_2d(mask, side_axis=side_axis, title=f"tooth_like debug")

        # 只检查前面的ruage
        if geom.inner_product(side_axis, self.orienter.forward) > 0.5 :
            if not self._two_sided_with_separation(faces, side_axis):
                return False, "two_sided_normals"

        return True, None


    @LazyAttribute
    def global_scale(self):
        curv = self.mesh.tri2tri_edge_curvatures
        x = -curv[np.isfinite(curv)]  # 负曲率强度候选（把符号翻成正）
        if x.size == 0:
            return 1.0
        else:
            return np.quantile(x, 0.9)  # 或 0.95

    def _boundary_crease_ratio(self, inside_mask, tau=0.0, use_magnitude=False):
        """
        inside_mask: bool, shape (n_face,)
        tau: 负曲率阈值（curv < -tau 认为是“沟”）
             tau=0.0 表示只要 <0 就算
        use_magnitude: 是否返回更稳定的 conf = neg_ratio * mean(-curv_neg)
                       如果 False，就只返回 neg_ratio

        return:
            neg_ratio, conf, stats_dict
        """
        nb = self.mesh.face_neighbors  # (n_face, 3), -1 表示无邻居（开边）
        curv = self.mesh.tri2tri_edge_curvatures  # (n_face, 3), 你说的 face-to-face 边曲率
        curv_min = np.min(curv)

        inside = inside_mask
        # 只在 inside 的 face 上看 3 条邻边
        faces = np.where(inside)[0]
        assert faces.size > 0

        nb_f = nb[faces]  # (m,3)
        curv_f = curv[faces]  # (m,3)

        # 邻居有效（不是 -1）
        valid_nb = (nb_f != -1)

        # 邻居是否在 inside（注意：nb_f 里有 -1，先用 valid_nb 过滤再索引）
        nb_in = np.zeros_like(valid_nb, dtype=bool)
        nb_in[valid_nb] = inside[nb_f[valid_nb]]

        # 边界边：inside face 的某条边，邻居存在且邻居不在 inside
        boundary_edge = valid_nb & (~nb_in)

        if not boundary_edge.any():
            return 0.0, 0.0

        bcurv = curv_f[boundary_edge]
        bcurv = bcurv[np.isfinite(bcurv)]
        assert bcurv.size > 0

        # # 负曲率边计数：curv < -tau
        # crease = bcurv < (-tau)
        # n_boundary = bcurv.size
        # n_crease = int(crease.sum())
        #
        # crease_ratio = float(n_crease / (n_boundary + 1e-12))
        #
        # if not use_magnitude:
        #     conf = crease_ratio
        #     mean_neg_mag = 0.0
        # else:
        #     # 负曲率强度：mean(-curv) over negative edges
        #     if n_crease > 0:
        #         mean_neg_mag = float((-bcurv[crease]).mean())
        #     else:
        #         mean_neg_mag = 0.0
        #     conf = float(crease_ratio * mean_neg_mag)

        neg = (-bcurv - tau)  # 把阈值扣掉，>0 才算沟，越大越沟
        neg = np.maximum(neg, 0.0)

        # 归一化并饱和
        edge_score = 1.0 - np.exp(-neg / (self.global_scale + 1e-12))  # [0,1)

        conf = float(edge_score.mean())  # 直接就是整体“沟置信度”
        # 如果你还想保留 ratio，也可以同时返回：
        crease_ratio = float((neg > 0).mean())

        return crease_ratio, conf

    def _pick_T_by_boundary_conf(self, peak, T_cap, T_lower=0.0, q_num=60, tau=0.0, use_magnitude=False,):
        MAX_WIDTH = 11.5
        MAX_HEIGHT = 10.0
        MAX_DROP = 4.0
        MAX_SLENDER = 1.7
        w_width = 0.2
        w_drop = 0.3
        w_slender = 0.2

        costs = self.peak_costs[peak]
        # 在 cap 内的可达区域
        valid = np.isfinite(costs) & (costs > 0) & (costs < T_cap)
        idx = np.where(valid)[0]
        # assert idx.size > 50

        c_sorted = np.sort(costs[idx])
        if T_lower > 0.0:
            # Plan C: 在 [T_lower, T_cap] 内均匀扫描候选 T。
            # 这个范围很窄(典型 [0.6*T*, T*]),用线性扫描即可,不需要分位数分布。
            if T_lower >= T_cap:
                return None, "T_lower >= T_cap"
            T_list = np.linspace(T_lower, T_cap, q_num)
        else:
            # 用分位数生成候选阈值(更均匀覆盖 cost 轴)
            qs = np.linspace(0.05, 0.95, q_num)
            T_list = np.quantile(c_sorted, qs)

        # # 最大可达面积（在 cap 内）
        # mask_cap = np.zeros(costs.shape[0], dtype=bool)
        # mask_cap[idx] = True
        # A_cap = float(self.mesh.area_faces[mask_cap].sum())
        # A_min = max(min_abs_area, min_area_frac * A_cap)

        # 参考 omin（用较小阈值下的区域作为“牙冠顶部附近”）
        T_ref = float(np.quantile(costs[valid], 0.20))
        mask_ref = (costs < T_ref)
        ref_stats = self._region_geom_stats(mask_ref)
        omin_ref = ref_stats["omin"]

        best = None
        for T in T_list:
            mask = (costs < T)
            geo_stats = self._region_geom_stats(mask)

            # ---几何形态学分析过滤---
            result, reason = self._is_tooth_like_region(mask, geo_stats, min_area=20, max_slender=MAX_SLENDER, omin_ref=omin_ref)
            # if reason == "two_sided_normals":
            #     return None, reason
            if not result:
                continue

            # --- 曲率边界置信度（你原始负曲率逻辑）---
            neg_ratio, conf = self._boundary_crease_ratio(mask, tau=tau, use_magnitude=use_magnitude)

            # --- 软惩罚（可选，进一步偏向“既完整又不冒险”）---
            # 这里把宽度、下探、细长条都做轻量惩罚
            penalty = 0.0
            penalty += w_width * (geo_stats["width"] / (MAX_WIDTH + 1e-9))
            penalty += w_slender * (geo_stats["slender"] / (MAX_SLENDER + 1e-9))
            penalty += w_drop * (max(0.0, (omin_ref - geo_stats["omin"])) / (MAX_DROP + 1e-9))
            score = float(conf - penalty)

            if best is None or score > best["score"]:
                best = {"T": float(T), "score": score, "conf": float(conf)}

        if best is None:
            return None, "No candidate passed"

        return best["T"], "Picked by conf with geom"

    def _estimate_peak_threshold_plus(self, peak, min_cap=0.6):
        spill_thr = self.spill_thresholds[peak]

        T_cap = 0.95 * float(spill_thr)
        if T_cap < min_cap:
            return None, "Spilled"

        # 用边界负曲率占比选阈值
        T_best, reason= self._pick_T_by_boundary_conf(peak=peak, T_cap=T_cap, q_num=70, tau=0.5,use_magnitude=True)
        return T_best, reason

    def _apply_per_peak_thresholds(self):
        for peak in self.peaks:
            T, reason = self._estimate_peak_threshold_plus(peak)
            if T is None:
                self.discarded_peaks[reason].add(peak)
                # peak.spilled = True
                continue

            self.peak_max_costs[peak] = T
            self.parse_spread(peak, T)

    def _apply_plan_c_thresholds(self, T_star, alpha=0.6, q_num=60):
        """方案 C: 以 Scheme 1 选出的全局 T_star 为上界,
        per-peak 在 [alpha * T_star, T_star] 内用边界 conf 准则微调 T_i。

        设计哲学:
        - T_star 是 coverage 最大化的产物,自带 anti-merging 反馈(向上扩张
          会让相邻牙合并被几何过滤拒,coverage 掉),所以"向上"方向已被 Scheme 1
          锁死,plan C 只允许向下收缩。
        - 收缩是为了修 Scheme 1 的"统一阈值偏高让某颗牙的区域被几何过滤拒"
          的失败模式。
        - 若 [alpha*T_star, T_star] 内没有候选通过准则,退回 T_star
          (= Scheme 1 对该 peak 的选择),保证不比 Scheme 1 差。
        """
        self.discarded_peaks.clear()
        self.discarded_overlap_groups.clear()
        self.discarded_teeth.clear()
        self.peak_max_costs.clear()

        T_lower = alpha * T_star

        for peak in self.peaks:
            peak.spilled = False
            T, _reason = self._pick_T_by_boundary_conf(
                peak=peak,
                T_cap=T_star,
                T_lower=T_lower,
                q_num=q_num,
                tau=0.5,
                use_magnitude=True,
            )
            if T is None:
                # 收缩区间内无候选通过 → 退回 Scheme 1 的 T_star
                T = T_star
            self.peak_max_costs[peak] = T
            self.parse_spread(peak, T)

    def try_with_max_cost(self, max_cost):
        self.max_cost = max_cost
        self.discarded_peaks.clear()
        self.discarded_overlap_groups.clear()
        self.discarded_teeth.clear()

        for peak in self.peaks:
            peak.spilled = False

        [self.parse_spread(peak, max_cost) for peak in self.peaks]

        self._build_quadratic()
        self._build_overlapping_area_groups()
        self._build_quadratic()
        for group in self.overlapping_area_groups:
            group.update_quadratic(self.quadratic)

        self._group_inline_area_groups()
        self._build_teeth()

        del self.tri2tri_costs
        return self.tooth_coverage()

    def parse_spread(self, peak, max_cost):
        accumulative_costs = self.peak_costs[peak]
        mask = accumulative_costs < max_cost

        # Test if the distance limit from the bound_costmap was actually used.
        peak.spilled = spilled = self.spill_thresholds[peak] < max_cost

        # If it was used then that peak's spread is labelled as "spilled".
        if spilled:
            # This should only happen if the peak was not on a tooth in the first place. Hence this
            # peak can be discarded.
            self.discarded_peaks["Spilled"].add(peak)

        self.peak_masks[peak] = mask

    def _spread_from_peaks(self):
        for peak in self.peaks:
            self._spread_from_peak(peak, self.peak_tri2tri_costs(peak))

    def _spread_from_peak(self, peak, costs):  # core region growing algorithm
        peak_triangles = np.where(self.mesh.faces == peak.index)[0]

        # init accumulative costs
        accumulative_cost = np.full(self.mesh.faces.shape[0], self.INF_COST, dtype=float)
        accumulative_cost[peak_triangles] = 0.0

        # init queue and shortest flags
        is_shortest = np.zeros(self.mesh.faces.shape[0], dtype=bool)  # 1 means the face's minimum accumulative cost
        faces_init = self.mesh.face_neighbors[peak_triangles].reshape(-1)  # has been got
        que = queue.PriorityQueue()
        [que.put((accumulative_cost[face], face)) for face in faces_init]

        # spread from triangles containing peak
        while not que.empty():
            face = que.get()[1]
            if is_shortest[face] == 1:
                continue
            else:
                is_shortest[face] = 1
                # make sure the region won't spread too widely
                # face_center = self.mesh.triangles_center[face]
                # width1 = abs(np.inner(self.orienter.right, face_center - self.mesh.vertices[peak_idx]))
                # width2 = abs(np.inner(self.orienter.forward, face_center - self.mesh.vertices[peak_idx]))
                # height = abs(np.inner(self.orienter.occlusal, face_center - self.mesh.vertices[peak_idx]))
                # if width1 > self._MAX_SPREAD_WIDTH or width2 > self._MAX_SPREAD_WIDTH or height > self._MAX_SPREAD_HEIGHT:
                #     # this peak beyond the max spreading range, discard it
                #     self.discarded_peaks['Spilled Peaks'].add(peak_idx)
                #     return
                #     # continue

                face_adj = self.mesh.face_neighbors[face]
                edges_cost_adj = costs[face]
                for i, face_ in enumerate(face_adj):
                    if face_ == -1:
                        continue
                    if accumulative_cost[face] + edges_cost_adj[i] < accumulative_cost[face_]:
                        accumulative_cost[face_] = accumulative_cost[face] + edges_cost_adj[i]
                        que.put((accumulative_cost[face_], face_))

        self.peak_costs[peak] = accumulative_cost
        self.peak_masks[peak] = is_shortest

    def _predict_spillage_thresholds(self):
        self.spill_thresholds = {}
        self.touches_edge_thresholds = {}
        old = np.seterr(invalid="ignore")
        tri_has_no_neighbour = self.mesh.face_neighbors == -1
        closed = not tri_has_no_neighbour.any()
        for (peak, costs) in self.peak_costs.items():
            self.spill_thresholds[peak] = spillage_threshold(costs, self.peak_tri2tri_costs(peak) >= self.INF_COST)
            # print(self.spill_thresholds[peak])
            if not closed:
                self.touches_edge_thresholds[peak] = spillage_threshold(costs, tri_has_no_neighbour)
        np.seterr(**old)

    def _build_overlapping_area_groups(self):
        """
        Find all peak spreads that share area on the mesh. The peak groups are stored in
        `self.overlapping_arg_groups`, an array of sets of args.

        It allows indirect groups. i.e. If peaks[0] overlaps with peaks[1] and peaks[1] overlaps
        with peaks[2] but peaks[0] doesn't overlap with peaks[2] then they are all grouped
        together anyway.
        """
        filtered_peaks = self.valid_peaks
        num_peak = len(filtered_peaks)

        # `overlap_adjacency_matrix` is a square bool array.
        # `overlap_adjacency_matrix[i, j]` = do peaks[i] and peaks[j] overlap?
        overlap_adjacency_matrix = np.zeros((num_peak, num_peak))
        for i in range(num_peak):
            overlap_adjacency_matrix[i][i] = 1
            mask1 = self.peak_masks[filtered_peaks[i]]
            for j in range(i + 1, num_peak):
                mask2 = self.peak_masks[filtered_peaks[j]]
                if np.bitwise_and(mask1, mask2).any():
                    overlap_adjacency_matrix[i][j] = 1
                    overlap_adjacency_matrix[j][i] = 1

        # build overlapping area groups
        self.overlapping_area_args = []
        self.overlapping_area_groups = []
        flag = np.zeros(num_peak, dtype=bool)
        for i in range(num_peak):
            if flag[i]:
                continue
            connected = np.where(overlap_adjacency_matrix[i] == 1)[0]
            group_peak_args = np.unique(np.where(overlap_adjacency_matrix[connected] == 1)[1])
            flag[group_peak_args] = True
            group_peaks = filtered_peaks[group_peak_args]
            mask = mask_or(*(self.peak_masks[i] for i in group_peaks))
            self.overlapping_area_args.append(group_peaks)
            area_group = OverlappingAreaGroup(group_peaks, mask, self.mesh, self.orienter, self.quadratic)

            valid = False
            if area_group.width > self.MAX_TOOTH_WIDTH:
                # Occasionally you get very long stretches of gum just beneath
                # the incisors.
                self.discarded_overlap_groups["Too Wide"].append(area_group)
                # self.overlapping_area_groups.append(area_group)
            elif area_group.is_one_axis_dominate:
                # To be a tooth the area should have significant extent
                # in at least two axes.
                self.discarded_overlap_groups["One Axis Dominate"].append(area_group)
                # self.overlapping_area_groups.append(area_group)
            elif area_group.is_one_sided or \
                (geom.inner_product(area_group.buccal, self.orienter.forward) > 0.5 and not self._two_sided_with_separation(np.where(mask)[0], area_group.buccal)):
                # To be a cusp of a tooth the area should have both lingual
                # facing and buccal facing parts.
                self.discarded_overlap_groups["Only on One Side"].append(area_group)

                # self.overlapping_area_groups.append(area_group)
            else:
                self.overlapping_area_groups.append(area_group)
                valid = True

            if not valid:
                for peak in group_peaks:
                    self.discarded_peaks["In Invalid Group"].add(peak)

    def _build_quadratic(self):
        """Build the quadratic (approximation of the jaw line) fitting to the point of each peak
        that isn't `spilled`. Use the quadratic to sort and enumerate the peaks (including the
        spilled ones) by their position along the quadratic. Modify `self.peaks` and
        `self.peak_points` to reflect the reordering.
        """
        #peak_points_unspilled = np.array(self.mesh.vertices[self.valid_peaks])
        peak_points_unspilled = np.array([peak.point for peak in self.valid_peaks])

        assert len(peak_points_unspilled) >= 3, "Not enough valid peaks to build quadratic"
        self.quadratic = Quadratic3D(peak_points_unspilled, self.orienter)

        # This just tests "how tall is the quadratic?".
        ys = self.quadratic.quadratic_2d.points[:, 1]
        # assert self.quadratic.quadratic_2d.height > 1.0 * ys.std()
        """Least squares quadratic is a poor approximation of the jaw line. This
        typically happens if there are raised areas in the centre-rear of the
        model. Other than manually removing these areas, there is nothing that
        can be done to fix this."""

    def _group_inline_area_groups(self):
        """Next group overlapping_area_groups if they are inline to join the lingual and buccal
        cusps of molars/premolars. This is done by projecting the points in each area quadratic
        to get a 1D line of points. The range of each area's points is found and compared with
        the ranges from other areas to determine if they are inline. """

        # Overlap in this method refers to the ranges of the projections overlapping rather than
        # areas overlapping as it was before. Apart from `self.overlapping_area_groups`.

        # This method considers both the absolute width of a range overlap (mm) and the ratio of
        # overlap_width / min(width of each range).

        n = len(self.overlapping_area_groups)
        assert n > 0

        overlap_width_map = np.zeros((n, n))
        overlap_ratio_map = np.zeros((n, n))

        # Cycle through all possible pairs, recording the width and ratios in
        # the above square arrays.
        for i in range(n):
            area_group_i = self.overlapping_area_groups[i]
            for j in range(i + 1, n):
                area_group_j = self.overlapping_area_groups[j]

                # Skip if they are more than a tooth's width apart.
                # This is approximated lazily by looking at the last and first peak of each group
                # peak_point1 = self.mesh.vertices[max(area_group_i.peaks)]
                # peak_point2 = self.mesh.vertices[min(area_group_j.peaks)]
                if geom.magnitude(area_group_i.peaks[-1].point - area_group_j.peaks[0].point) > self.MAX_TOOTH_WIDTH:
                    continue

                # Skip if they are too far apart in the occlusal direction.
                if abs(inner_product(area_group_i.obb.center - area_group_j.obb.center, self.orienter.occlusal)) > 3:
                    continue

                # The actual maths is handled in `OverlappingAreasGroup.get_inline_overlap`
                overlap = area_group_i.get_inline_overlap(area_group_j)
                overlap_width_map[i, j] = overlap.width
                overlap_ratio_map[i, j] = overlap.ratio

        # Magic made up rule that combines all the above into a hard "inline or not inline" square
        # bool array.
        mask = ((12 > overlap_width_map) & (overlap_width_map > 2) & (overlap_ratio_map > 0.55))

        # Again convert bool array to arg groups
        self.inline_group_args = Grouping(mask).groups

    def _build_teeth(self):
        """Convert each group from `self.inline_group_args` to a Tooth instance from tooth_class.py.
        Also filters away instances that cover too little area to be a tooth. Otherwise you get tiny
        little isolated bumps which are irrelevant. """

        self.teeth = []
        for (i, args) in enumerate(self.inline_group_args):
            # Each tooth receives all the OverlappingAreaGroup objects from an inline group. We
            # don't know which tooth is which yet so each is given an enumeration as a convenient ID.
            groups = [self.overlapping_area_groups[j] for j in sorted(args)]
            tooth = Tooth(groups, i + 1)
            if tooth.area < self.MIN_TOOTH_AREA:
                self.discarded_teeth["Area too small"].append(tooth)
                continue

            self.teeth.append(tooth)

    def test_max_costs(self, max_costs=None):
        if max_costs is None:
            max_costs = np.logspace(np.log10(0.3), np.log10(2), 20)

        max_costs = iter(max_costs)
        self.max_cost_results = []

        # first loop: find first successful max_cost
        for i in max_costs:
            try:
                self.max_cost_results.append((i, self.try_with_max_cost(i)))
                break
            except RuntimeError as ex:
                print(ex)
                self.max_cost_results.append((i, np.nan))

        # second loop: continue until first failure
        for i in max_costs:
            try:
                self.max_cost_results.append((i, self.try_with_max_cost(i)))
            except RuntimeError as ex:
                print(ex)
                self.max_cost_results.append((i, np.nan))
                break

        self.max_cost_results = np.array(self.max_cost_results)

    def pick_optimal_max_cost(self):
        try:
            max_cost = self.max_cost_results[np.nanargmax(self.max_cost_results[:, 1]), 0]
        except ValueError:
            raise RuntimeError(
                "ALR couldn't create any possibly valid partitioning of the model."
                "Please check that your model looks at least vaguely like a dental model"
                " and report it if it does.")
        self.try_with_max_cost(max_cost)


    def tooth_coverage(self):
        if len(self.teeth) == 0:
            raise RuntimeError("No teeth found.")
        return mask_or(i.mask for i in self.teeth).sum()


        









