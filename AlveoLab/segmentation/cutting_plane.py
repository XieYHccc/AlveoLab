import numpy as np
import trimesh as tm
from typing import Dict, List, Optional, Tuple

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector.")
    return v / n


def _extract_plane_components_as_vertex_loops(
    mesh: tm.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> List[np.ndarray]:
    """
    返回该平面与mesh交线的连通分量，每个分量是 polyline 点序列 (m_i, 3)。
    使用 trimesh.section -> Path3D.discrete 获取。
    """
    sec = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if sec is None:
        return []

    # Path3D.discrete: list of polylines, each (k,3)
    loops = []
    for pl in sec.discrete:
        pl = np.asarray(pl, dtype=float)
        if pl.shape[0] >= 3:
            loops.append(pl)
    return loops


def _is_closed_polyline(poly: np.ndarray, close_tol: float = 1e-3) -> bool:
    if poly.shape[0] < 3:
        return False
    return np.linalg.norm(poly[0] - poly[-1]) <= close_tol


def _nearest_vertex_ids(mesh: tm.Trimesh, points: np.ndarray) -> np.ndarray:
    """
    将交线采样点映射到最近mesh顶点索引（用于取顶点曲率）。
    """
    # trimesh.proximity.closest_point 可返回最近三角面点，但这里要顶点id更直接用KDTree
    # trimesh内置kdtree可用
    kdtree = mesh.kdtree
    d, vid = kdtree.query(points)
    return np.asarray(vid, dtype=np.int64)


def _variance_energy(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float).reshape(-1)
    if vals.size <= 1:
        return np.inf
    m = vals.mean()
    return float(((vals - m) ** 2).sum() / (vals.size - 1))


def _curvature_on_loop_vertices(
    mesh_wrap,                         # 你的 Mesh 封装对象（有 _mesh / vertex_mean_curvature 等）
    loop_vid: np.ndarray,
    curvature_mode: str = "kmin",      # "kmin" or "mean"
    kmin_array: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    loop_vid 对应交线最近顶点索引，返回该环上的曲率值数组。
    """
    if curvature_mode == "kmin":
        # 你如果已有最小主曲率数组，直接传 kmin_array（长度=|V|）
        if kmin_array is None:
            raise ValueError("curvature_mode='kmin' requires kmin_array.")
        return np.asarray(kmin_array, dtype=float)[loop_vid]

    elif curvature_mode == "mean":
        # 用你Mesh类的懒加载均值曲率
        mean_c = np.asarray(mesh_wrap.vertex_mean_curvature, dtype=float).reshape(-1)
        return mean_c[loop_vid]

    else:
        raise ValueError(f"Unknown curvature_mode: {curvature_mode}")


def find_optimal_gingiva_plane_trimesh(
    mesh_wrap,                         # 你的 Mesh 对象（或兼容：有._mesh）
    occlusal: np.ndarray,
    step_mm: float = 1.0,              # 论文经验值约1mm
    top_margin_mm: float = 3.0,
    delta_down_mm: float = 1.2,        # 额外向下微调
    close_tol: float = 1e-3,
    curvature_mode: str = "kmin",      # "kmin" / "mean"
    kmin_array: Optional[np.ndarray] = None,
    min_loop_points: int = 20,
) -> Tuple[float, np.ndarray, Dict]:
    """
    返回:
      z_final: 最终切割高度（沿occlusal方向的标量）
      gingiva_ring_vidx: 最终切割平面交线最近顶点索引（去重）
      dbg: 调试信息（每一步候选记录）
    """
    tm_mesh = mesh_wrap._mesh if hasattr(mesh_wrap, "_mesh") else mesh_wrap
    n = _normalize(occlusal)

    V = np.asarray(tm_mesh.vertices, dtype=float)
    h = V @ n

    # 从靠近顶面开始往下扫
    z_top = float(np.max(h) - top_margin_mm)
    z_bottom = float(np.min(h) - 1e-6)

    # 记录
    dbg_steps = []
    one_loop_started = False
    best_z = None
    best_loop_vid = None

    prev_norm_e = None
    energies_raw = []
    candidate_records = []

    z = z_top
    while z >= z_bottom:
        origin = n * z
        loops = _extract_plane_components_as_vertex_loops(tm_mesh, origin, n)

        # 只保留“闭合且点数足够”的环
        closed_loops = [lp for lp in loops if _is_closed_polyline(lp, close_tol) and lp.shape[0] >= min_loop_points]

        rec = {
            "z": z,
            "n_loops": len(closed_loops),
            "energy": None,
            "accepted": False,
        }

        if len(closed_loops) == 1:
            one_loop_started = True

            loop_pts = closed_loops[0]
            loop_vid = _nearest_vertex_ids(tm_mesh, loop_pts)
            loop_vid = np.unique(loop_vid)

            if loop_vid.size >= 3:
                cvals = _curvature_on_loop_vertices(
                    mesh_wrap=mesh_wrap,
                    loop_vid=loop_vid,
                    curvature_mode=curvature_mode,
                    kmin_array=kmin_array
                )
                e = _variance_energy(cvals)
                rec["energy"] = e
                energies_raw.append(e)
                candidate_records.append((z, loop_vid, e))
                rec["accepted"] = True

        dbg_steps.append(rec)

        # 一旦进入单环阶段后又失去交线，按论文思路用“最后有效单环”兜底
        if one_loop_started and len(closed_loops) == 0:
            break

        z -= step_mm

    if len(candidate_records) == 0:
        raise RuntimeError("No valid one-loop intersections found. Check occlusal direction / step / model quality.")

    # 对单环能量做线性归一化到[0,1]
    e_arr = np.array([r[2] for r in candidate_records], dtype=float)
    e_min, e_max = float(e_arr.min()), float(e_arr.max())
    if e_max - e_min < 1e-12:
        e_norm = np.zeros_like(e_arr)
    else:
        e_norm = (e_arr - e_min) / (e_max - e_min)

    # 论文启发：在 e_norm < 0.5 区间内，找“停止下降”的点
    # 实现：从上到下扫描，若当前>=前一项(允许小容差)则停止，取前一项（最后一个下降点）
    pick_idx = None
    prev = None
    for i, en in enumerate(e_norm):
        if en < 0.5:
            if prev is not None and en >= prev - 1e-6:
                pick_idx = max(i - 1, 0)
                break
            prev = en

    if pick_idx is None:
        # 若没触发“停止下降”，用 e_norm<0.5 的最后一个；再不行用全局最小
        valid = np.where(e_norm < 0.5)[0]
        if len(valid) > 0:
            pick_idx = int(valid[-1])
        else:
            pick_idx = int(np.argmin(e_norm))

    z_pick, loop_vid_pick, _ = candidate_records[pick_idx]
    z_final = float(z_pick - delta_down_mm)

    # 用 z_final 再求一次交线顶点约束（若失败则回退 pick）
    loops_final = _extract_plane_components_as_vertex_loops(tm_mesh, n * z_final, n)
    closed_final = [lp for lp in loops_final if _is_closed_polyline(lp, close_tol) and lp.shape[0] >= min_loop_points]
    if len(closed_final) >= 1:
        # 如果>1个环，取点数最多的那个
        loop_final = max(closed_final, key=lambda x: x.shape[0])
        vid_final = np.unique(_nearest_vertex_ids(tm_mesh, loop_final))
    else:
        vid_final = loop_vid_pick

    dbg = {
        "steps": dbg_steps,
        "candidate_records_count": len(candidate_records),
        "candidate_z": [float(r[0]) for r in candidate_records],
        "candidate_energy_raw": e_arr.tolist(),
        "candidate_energy_norm": e_norm.tolist(),
        "picked_idx": int(pick_idx),
        "picked_z_before_delta": float(z_pick),
        "z_final": z_final,
        "curvature_mode": curvature_mode,
    }

    return z_final, vid_final.astype(np.int64), dbg
