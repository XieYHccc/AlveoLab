import numpy as np
import trimesh as tm
from typing import Dict, List, Tuple


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector.")
    return v / n


def _extract_plane_components_as_vertex_loops(
    mesh: tm.Trimesh, plane_origin: np.ndarray, plane_normal: np.ndarray
) -> List[np.ndarray]:
    sec = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if sec is None:
        return []
    loops = []
    for pl in sec.discrete:
        pl = np.asarray(pl, dtype=float)
        if pl.ndim == 2 and pl.shape[0] >= 3 and pl.shape[1] == 3:
            loops.append(pl)
    return loops


def _is_closed_polyline(poly: np.ndarray, close_tol: float = 1e-3) -> bool:
    return poly.shape[0] >= 3 and np.linalg.norm(poly[0] - poly[-1]) <= close_tol


def _nearest_vertex_ids(mesh: tm.Trimesh, points: np.ndarray) -> np.ndarray:
    _, vid = mesh.kdtree.query(points)
    return np.asarray(vid, dtype=np.int64)


def _variance_energy(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float).reshape(-1)
    if vals.size <= 1:
        return np.inf
    m = vals.mean()
    return float(((vals - m) ** 2).sum() / (vals.size - 1))

def _pick_first_decreasing_then_local_min(
    z_arr: np.ndarray,
    e_norm: np.ndarray,
    start_drop_eps: float = 1e-4,
    rise_eps: float = 1e-4,
    min_consecutive_drop: int = 1,
) -> int:
    """
    规则：
    1) 从头扫描，先找到“第一次进入递减趋势”（至少 min_consecutive_drop 次连续下降）
    2) 从该处继续向后，直到第一次由降转升（上升超过 rise_eps）
    3) 返回该转折前的局部最小位置
    4) 若未找到转升，则返回这段递减尾部最小值
    5) 若一直没进入递减趋势，退化为全局最小
    """
    e = np.asarray(e_norm, dtype=float).reshape(-1)
    n = e.size
    if n == 0:
        raise ValueError("Empty energy array.")
    if n == 1:
        return 0

    # 一阶差分：d[i] = e[i] - e[i-1], i=1..n-1
    d = np.diff(e)

    # ---------- 找首次“稳定递减”起点 ----------
    start_idx = None
    consec = 0
    for i in range(1, n):
        if d[i - 1] < -start_drop_eps:
            consec += 1
            if consec >= min_consecutive_drop:
                # 递减段开始位置（回到这段第一个下降点的左端）
                start_idx = i - min_consecutive_drop
                break
        else:
            consec = 0

    # 没找到递减趋势 -> 退化全局最小
    if start_idx is None:
        return int(np.argmin(e))

    # ---------- 从递减段开始，找“首次由降转升” ----------
    valley_idx = start_idx
    best_val = e[start_idx]
    # 先向后追踪最小值
    for i in range(start_idx + 1, n):
        if abs(z_arr[i] - z_arr[i - 1]) > 2:
            return int(valley_idx)
        if e[i] < best_val:
            best_val = e[i]
            valley_idx = i

        # 一旦出现明显上升，且当前点在最小值之后，认为谷底已过
        if (e[i] - e[i - 1]) > rise_eps and i > valley_idx:
            return int(valley_idx)

    # 若后面没出现明显上升，取递减后段中最小
    return int(valley_idx)

def find_optimal_gingiva_plane_trimesh_mean_paperlike(
    mesh_wrap,
    occlusal: np.ndarray,
    step_mm: float = 1.0,
    top_margin_mm: float = 3.0,
    close_tol: float = 1e-3,
    min_loop_points: int = 20,
) -> Tuple[float, np.ndarray, Dict]:
    tm_mesh = mesh_wrap._mesh if hasattr(mesh_wrap, "_mesh") else mesh_wrap
    n = _normalize(occlusal)

    V = np.asarray(tm_mesh.vertices, dtype=float)
    h = V @ n
    z_top = float(np.max(h) - top_margin_mm)
    z_bottom = float(np.min(h) - 1e-6)

    mean_curv = np.asarray(mesh_wrap.vertex_minimum_curvature, dtype=float).reshape(-1)
    if mean_curv.shape[0] != V.shape[0]:
        raise ValueError("vertex_mean_curvature length mismatch with mesh vertices.")

    candidate = []
    dbg_steps = []
    one_loop_started = False

    z = z_top
    while z >= z_bottom:
        loops = _extract_plane_components_as_vertex_loops(tm_mesh, n * z, n)
        closed_loops = [lp for lp in loops if _is_closed_polyline(lp, close_tol) and lp.shape[0] >= min_loop_points]

        rec = {"z": float(z), "n_loops": int(len(closed_loops)), "accepted": False, "energy": None}

        if len(closed_loops) == 1:
            one_loop_started = True
            loop_vid = np.unique(_nearest_vertex_ids(tm_mesh, closed_loops[0]))
            if loop_vid.size >= 3:
                e = _variance_energy(mean_curv[loop_vid])
                candidate.append((float(z), loop_vid, float(e)))
                rec["accepted"] = True
                rec["energy"] = float(e)

        dbg_steps.append(rec)

        if one_loop_started and len(closed_loops) == 0:
            break

        z -= step_mm

    if len(candidate) == 0:
        raise RuntimeError("No valid single-loop candidates found.")

    z_arr = np.array([c[0] for c in candidate], dtype=float)
    e_arr = np.array([c[2] for c in candidate], dtype=float)

    e_min, e_max = float(e_arr.min()), float(e_arr.max())
    if e_max - e_min < 1e-12:
        e_norm = np.zeros_like(e_arr)
    else:
        e_norm = (e_arr - e_min) / (e_max - e_min)

    # 关键修改：第一次进入递减后的局部最小
    pick_idx = _pick_first_decreasing_then_local_min(
        z_arr,
        e_norm,
        start_drop_eps=1e-4,
        rise_eps=1e-4,
        min_consecutive_drop=2
    )

    # pick_idx += 1
    z_final = float(z_arr[pick_idx])
    loop_vid_pick = candidate[pick_idx][1]

    loops_final = _extract_plane_components_as_vertex_loops(tm_mesh, n * z_final, n)
    closed_final = [lp for lp in loops_final if _is_closed_polyline(lp, close_tol) and lp.shape[0] >= min_loop_points]
    if len(closed_final) == 1:
        vid_final = np.unique(_nearest_vertex_ids(tm_mesh, closed_final[0]))
    else:
        vid_final = loop_vid_pick

    dbg = {
        "z_top": float(z_top),
        "z_bottom": float(z_bottom),
        "step_mm": float(step_mm),
        "candidate_count": int(len(candidate)),
        "candidate_z": z_arr.tolist(),
        "candidate_energy_raw": e_arr.tolist(),
        "candidate_energy_norm": e_norm.tolist(),
        "picked_idx": int(pick_idx),
        "z_final": float(z_final),
        "paper_like_except_curvature": "mean_curvature",
        "pick_rule": "first_decreasing_then_first_local_min",
    }

    return z_final, vid_final.astype(np.int64), dbg
