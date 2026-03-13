import numpy as np
import trimesh as tm
from collections import deque

def seed_faces_from_peaks(mesh: tm.Trimesh, peak_vidx: np.ndarray) -> np.ndarray:
    peak_vidx = np.asarray(peak_vidx, dtype=np.int64)
    vf = mesh.vertex_faces[peak_vidx].reshape(-1)
    vf = vf[vf >= 0]
    return np.unique(vf)

def extract_region_by_wall_faces(
    mesh: tm.Trimesh,
    wall_faces: np.ndarray,
    seed_faces: np.ndarray,
    add_wall_back: bool = True,
):
    """
    用 wall_faces 作为“不可穿越面带”，从 seed_faces 泛洪得到一侧区域。

    返回：
      region_faces: (K,) 面id
      region_mask: (F,) bool
    """
    F = mesh.faces.shape[0]
    wall_faces = np.unique(np.asarray(wall_faces, dtype=np.int64))
    wall_mask = np.zeros(F, dtype=bool)
    wall_mask[wall_faces] = True

    seed_faces = np.unique(np.asarray(seed_faces, dtype=np.int64))
    seed_faces = seed_faces[~wall_mask[seed_faces]]  # 种子落在墙上就剔除
    if seed_faces.size == 0:
        return np.array([], dtype=np.int64), np.zeros(F, dtype=bool)

    # 建 face 邻接表（dual graph）
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)  # (A,2)
    nbrs = [[] for _ in range(F)]
    for a, b in adj:
        nbrs[a].append(b)
        nbrs[b].append(a)

    visited = np.zeros(F, dtype=bool)
    dq = deque(seed_faces.tolist())
    visited[seed_faces] = True

    while dq:
        f = dq.popleft()
        for g in nbrs[f]:
            if visited[g]:
                continue
            if wall_mask[g]:          # 关键：不进入墙面
                continue
            visited[g] = True
            dq.append(g)

    region_mask = visited.copy()

    if add_wall_back and wall_faces.size > 0:
        # 把“紧贴 region 一侧”的墙面补回去：只加那些与 region 相邻的 wall faces
        # （这样不会把另一侧的墙面也全加进来）
        wall_keep = np.zeros_like(wall_mask)
        for wf in wall_faces:
            # 只要墙面有邻居在 region，就把它并回去
            for nb in nbrs[wf]:
                if region_mask[nb]:
                    wall_keep[wf] = True
                    break
        region_mask |= wall_keep

    region_faces = np.where(region_mask)[0]
    return region_faces, region_mask

def extract_tooth_submesh_from_candidate_faces(
    mesh,
    candidate,                 # LoopCandidate
    tooth_peak_vidx: np.ndarray,
):
    wall_faces = np.asarray(candidate.face_ids, dtype=np.int64)
    seed_faces = seed_faces_from_peaks(mesh, tooth_peak_vidx)

    region_faces, region_mask = extract_region_by_wall_faces(
        mesh, wall_faces=wall_faces, seed_faces=seed_faces, add_wall_back=True
    )
    if region_faces.size == 0:
        return None, region_mask

    tooth_tm = mesh.submesh([region_faces], append=True, repair=False)
    return tooth_tm, region_mask