import numpy as np
import trimesh
from scipy.spatial import cKDTree, ConvexHull


def _polyline_length(P: np.ndarray) -> float:
    if P.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(P[1:] - P[:-1], axis=1)))


def _is_closed(P: np.ndarray, close_tol: float) -> bool:
    if P.shape[0] < 3:
        return False
    return np.linalg.norm(P[0] - P[-1]) < close_tol


def _project_uv(P3: np.ndarray, right: np.ndarray, forward: np.ndarray) -> np.ndarray:
    right = np.asarray(right, dtype=float)
    forward = np.asarray(forward, dtype=float)
    return np.c_[P3 @ right, P3 @ forward]


def _polygon_area_2d(P2: np.ndarray) -> float:
    """Shoelace; P2 should be closed-ish, but we handle open."""
    if P2.shape[0] < 3:
        return 0.0
    x = P2[:, 0]
    y = P2[:, 1]
    # close
    x2 = np.r_[x, x[0]]
    y2 = np.r_[y, y[0]]
    return 0.5 * float(np.abs(np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1])))


def find_optimal_gingiva_plane_trimesh(
    mesh: trimesh.Trimesh,
    occlusal: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    step_mm: float = 0.5,
    top_margin_mm: float = 3.0,
    delta_down_mm: float = 1.2,
    window_mm: float = 4.0,
    stable_ratio: float = 0.75,
    close_tol_mm: float = 2.0,
    min_points: int = 30,
    min_length_mm: float = 20.0,
    # 面积阈值相对于“整口投影凸包面积”的比例（经验值，可根据数据调）
    min_area_frac: float = 0.03,
    max_area_frac: float = 0.60,
    # 选点密度：把环上点映射到最近顶点后，做一个下采样步长（越小点越密）
    sample_every_n: int = 2,
):
    """
    返回：
      z_final: 选中的最终平面高度（沿 occlusal 的坐标）
      ring_vidx: 作为 gingiva constraints 的顶点索引(np.ndarray)
      debug: dict（含扫描记录、最佳环 polyline 等，便于调试）
    """
    occlusal = np.asarray(occlusal, dtype=float)
    occlusal = occlusal / (np.linalg.norm(occlusal) + 1e-12)

    V = np.asarray(mesh.vertices, dtype=float)
    heights = V @ occlusal  # (N,)

    z_max = float(np.max(heights))
    z_min = float(np.percentile(heights, 20))  # 下限别太低，避免根部噪声
    z_start = z_max - top_margin_mm
    z_end = z_min

    if z_start <= z_end + 1e-6:
        raise RuntimeError("扫描范围无效：z_start <= z_end。检查 top_margin_mm 或模型方向。")

    # 估计整口投影面积（用于剔除过小/过大的环）
    uv_all = _project_uv(V, right, forward)
    hull = ConvexHull(uv_all)
    mouth_area = float(hull.volume)  # 2D ConvexHull.volume = area
    A_min = min_area_frac * mouth_area
    A_max = max_area_frac * mouth_area

    # 扫描高度（从上往下）
    zs = np.arange(z_start, z_end, -abs(step_mm), dtype=float)
    records = []  # 每层：{"z":..., "K":..., "best_loop":..., "len":..., "area":...}

    close_tol = close_tol_mm
    for z in zs:
        plane_origin = occlusal * z
        section = mesh.section(plane_origin=plane_origin, plane_normal=occlusal)
        if section is None:
            records.append({"z": z, "K": 0, "best_loop": None, "len": 0.0, "area": 0.0})
            continue

        # section.discrete: list of polylines (N_i,3)
        polylines = section.discrete
        loops = []
        for P in polylines:
            if P is None:
                continue
            P = np.asarray(P, dtype=float)
            if P.shape[0] < min_points:
                continue
            if not _is_closed(P, close_tol=close_tol):
                continue
            L = _polyline_length(P)
            if L < min_length_mm:
                continue
            uv = _project_uv(P, right, forward)
            A = _polygon_area_2d(uv)
            if A < A_min or A > A_max:
                continue
            loops.append((P, L, A))

        K = len(loops)
        if K == 0:
            records.append({"z": z, "K": 0, "best_loop": None, "len": 0.0, "area": 0.0})
            continue

        # 若有多个环：优先取“面积最大”的那个作为代表（通常是主环）
        loops.sort(key=lambda x: x[2], reverse=True)
        P_best, L_best, A_best = loops[0]
        records.append({"z": z, "K": K, "best_loop": P_best, "len": L_best, "area": A_best})

    # 选最优 z：找 K==1 的稳定窗口
    if not records:
        raise RuntimeError("没有任何截面记录。")

    zs_arr = np.array([r["z"] for r in records], dtype=float)
    K_arr = np.array([r["K"] for r in records], dtype=int)
    L_arr = np.array([r["len"] for r in records], dtype=float)
    A_arr = np.array([r["area"] for r in records], dtype=float)

    half_w = window_mm * 0.5

    best_score = -1e18
    best_idx = None

    for i, r in enumerate(records):
        if r["K"] != 1 or r["best_loop"] is None:
            continue

        z = r["z"]
        # 窗口内索引
        in_win = np.where((zs_arr <= z + half_w) & (zs_arr >= z - half_w))[0]
        if in_win.size < 3:
            continue

        stability = float(np.mean(K_arr[in_win] == 1))
        if stability < stable_ratio:
            continue

        # 平滑性：长度/面积抖动小更好
        Lw = L_arr[in_win]
        Aw = A_arr[in_win]
        L_mean = float(np.mean(Lw[Lw > 0]))
        A_mean = float(np.mean(Aw[Aw > 0]))
        L_std = float(np.std(Lw[Lw > 0])) if np.any(Lw > 0) else 1e9
        A_std = float(np.std(Aw[Aw > 0])) if np.any(Aw > 0) else 1e9

        # 偏好：更紧致（长度小）但别太极端；用对数温和化
        tight_bonus = -np.log((r["len"] / (np.min(Lw[Lw > 0]) + 1e-12)) + 1e-12)

        # 综合评分：稳定性为主，抖动惩罚，紧致奖励
        score = (
            5.0 * stability
            - 1.5 * (L_std / (L_mean + 1e-12))
            - 1.0 * (A_std / (A_mean + 1e-12))
            + 0.6 * tight_bonus
        )

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        # 兜底：如果没有稳定窗口，退化为“第一个出现 K==1 的高度”
        first_one = np.where(K_arr == 1)[0]
        if first_one.size == 0:
            raise RuntimeError("扫描范围内没有出现单环截面。请调高 z_end 或放宽过滤条件。")
        best_idx = int(first_one[0])

    z_star = float(records[best_idx]["z"])
    z_final = z_star - float(delta_down_mm)

    # 在 z_final 再截一次，取当时“最合理”的环
    plane_origin = occlusal * z_final
    section = mesh.section(plane_origin=plane_origin, plane_normal=occlusal)
    if section is None:
        raise RuntimeError("在 z_final 处无法得到截面。尝试减小 delta_down_mm 或扩大扫描范围。")

    polylines = section.discrete
    best_loop = None
    best_area = -1.0
    for P in polylines:
        if P is None:
            continue
        P = np.asarray(P, dtype=float)
        if P.shape[0] < min_points:
            continue
        if not _is_closed(P, close_tol=close_tol):
            continue
        L = _polyline_length(P)
        if L < min_length_mm:
            continue
        uv = _project_uv(P, right, forward)
        A = _polygon_area_2d(uv)
        if A < A_min or A > A_max:
            continue
        if A > best_area:
            best_area = A
            best_loop = P

    if best_loop is None:
        raise RuntimeError("z_final 处没有通过过滤的闭合环。请放宽 min_length/min_area 或调整 delta_down。")

    # loop points -> nearest vertex indices
    P = best_loop
    if sample_every_n > 1:
        P = P[:: int(sample_every_n)]

    kdt = cKDTree(V)
    _, idx = kdt.query(P, k=1)
    ring_vidx = np.unique(idx.astype(np.int64))

    debug = {
        "records": records,
        "best_score": best_score,
        "z_star": z_star,
        "z_final": z_final,
        "best_loop_poly3d": best_loop,
        "mouth_area_uv_hull": mouth_area,
        "A_min": A_min,
        "A_max": A_max,
    }

    return z_final, ring_vidx, debug
