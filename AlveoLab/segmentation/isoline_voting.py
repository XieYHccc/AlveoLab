import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


# ---------------------------
# marching triangles: segments + face ids
# ---------------------------

def _interp(p0, p1, v0, v1, iso) -> np.ndarray:
    t = (iso - v0) / (v1 - v0)
    return p0 + t * (p1 - p0)


def extract_isoline_segments(V: np.ndarray, F: np.ndarray, phi: np.ndarray, iso: float):
    segs = []
    fids = []
    for fi, tri in enumerate(F):
        ids = tri
        vals = phi[ids]
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if iso < vmin or iso > vmax or (vmax - vmin) < 1e-12:
            continue

        pts = []
        for (a, b) in [(0, 1), (1, 2), (2, 0)]:
            ia, ib = ids[a], ids[b]
            va, vb = float(phi[ia]), float(phi[ib])

            # strict crossing
            if (va - iso) * (vb - iso) < 0.0:
                pts.append(_interp(V[ia], V[ib], va, vb, iso))
            # vertex on iso (helps connectivity)
            elif abs(va - iso) < 1e-12 and abs(vb - iso) >= 1e-12:
                pts.append(V[ia])
            elif abs(vb - iso) < 1e-12 and abs(va - iso) >= 1e-12:
                pts.append(V[ib])

        if len(pts) < 2:
            continue

        if len(pts) == 2:
            segs.append([pts[0], pts[1]])
            fids.append(fi)
        else:
            for i in range(0, len(pts) - 1, 2):
                segs.append([pts[i], pts[i + 1]])
                fids.append(fi)

    if not segs:
        return np.zeros((0, 2, 3), dtype=np.float64), np.zeros((0,), dtype=np.int32)

    return np.asarray(segs, dtype=np.float64), np.asarray(fids, dtype=np.int32)


# ---------------------------
# stitching that preserves segment membership (for voting)
# ---------------------------

def stitch_segments_to_polylines_with_ids(
    segments: np.ndarray,
    seg_face_ids: np.ndarray,
    tol: float = 1e-4
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns list of components:
      (poly3d (N,3), face_ids (M,), seg_lens (M,))
    where M is number of segments in that polyline component.
    """
    if segments.shape[0] == 0:
        return []

    seg_lens = np.linalg.norm(segments[:, 1, :] - segments[:, 0, :], axis=1)

    def key(p):
        return tuple(np.round(p / tol).astype(np.int64))

    endpoint_map: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = {}
    for si in range(segments.shape[0]):
        endpoint_map.setdefault(key(segments[si, 0]), []).append((si, 0))
        endpoint_map.setdefault(key(segments[si, 1]), []).append((si, 1))

    used = np.zeros(segments.shape[0], dtype=bool)
    comps = []

    for start_si in range(segments.shape[0]):
        if used[start_si]:
            continue

        used[start_si] = True
        chain_pts = [segments[start_si, 0].copy(), segments[start_si, 1].copy()]
        chain_seg_ids = [start_si]

        def extend(end_point, forward=True):
            nonlocal chain_pts, chain_seg_ids
            while True:
                k = key(end_point)
                cands = endpoint_map.get(k, [])
                nxt = None
                for (si, end_id) in cands:
                    if used[si]:
                        continue
                    nxt = (si, end_id)
                    break
                if nxt is None:
                    break

                si, end_id = nxt
                used[si] = True
                a = segments[si, 0]
                b = segments[si, 1]
                next_pt = b if end_id == 0 else a

                if forward:
                    chain_pts.append(next_pt)
                    chain_seg_ids.append(si)
                    end_point = next_pt
                else:
                    chain_pts.insert(0, next_pt)
                    chain_seg_ids.insert(0, si)
                    end_point = next_pt

        extend(chain_pts[-1], forward=True)
        extend(chain_pts[0], forward=False)

        seg_ids = np.asarray(chain_seg_ids, dtype=np.int64)
        comps.append((
            np.asarray(chain_pts, dtype=np.float64),
            seg_face_ids[seg_ids],
            seg_lens[seg_ids],
        ))

    return comps


def polyline_is_closed(poly3d: np.ndarray, tol: float = 2e-3) -> bool:
    return np.linalg.norm(poly3d[0] - poly3d[-1]) < tol


# ---------------------------
# face weight g_i: ||grad phi|| per face
# ---------------------------

def triangle_grad_scalar(V: np.ndarray, tri: np.ndarray, phi: np.ndarray) -> np.ndarray:
    i0, i1, i2 = tri
    p0, p1, p2 = V[i0], V[i1], V[i2]
    f0, f1, f2 = phi[i0], phi[i1], phi[i2]

    e1 = p1 - p0
    e2 = p2 - p0
    n = np.cross(e1, e2)
    area2 = np.linalg.norm(n)
    if area2 < 1e-12:
        return np.zeros(3, dtype=np.float64)

    denom = area2 ** 2
    grad_l0 = np.cross(n, (p2 - p1)) / denom
    grad_l1 = np.cross(n, (p0 - p2)) / denom
    grad_l2 = np.cross(n, (p1 - p0)) / denom
    return f0 * grad_l0 + f1 * grad_l1 + f2 * grad_l2


def compute_face_grad_magnitudes(V: np.ndarray, F: np.ndarray, phi: np.ndarray) -> np.ndarray:
    g = np.zeros(F.shape[0], dtype=np.float64)
    for fi, tri in enumerate(F):
        grad = triangle_grad_scalar(V, tri, phi)
        g[fi] = np.linalg.norm(grad)
    # normalize to [0,1]
    mn, mx = float(np.min(g)), float(np.max(g))
    if mx - mn < 1e-12:
        return np.zeros_like(g)
    return (g - mn) / (mx - mn)


# ---------------------------
# Dot-Scissor style scoring + voting
# ---------------------------

def gaussian_weight(k: int, sigma: float) -> float:
    return float(np.exp(-(k * k) / (2.0 * sigma * sigma)))


def stepped_concavity_over_ordered(lengths: np.ndarray, sigma: float = 2.0, K: int = 4) -> np.ndarray:
    """
    Δ_{i,k} = 2 r_i - r_{i-k} - r_{i+k}
    conc_i = Σ f(k)*Δ / Σ f(k)
    """
    n = len(lengths)
    conc = np.zeros(n, dtype=np.float64)
    for i in range(n):
        num = 0.0
        den = 0.0
        for k in range(1, K + 1):
            if i - k < 0 or i + k >= n:
                continue
            w = gaussian_weight(k, sigma)
            num += w * (2.0 * lengths[i] - lengths[i - k] - lengths[i + k])
            den += w
        conc[i] = num / (den + 1e-12)
    return conc


def g_squash(x: np.ndarray) -> np.ndarray:
    """Dot-Scissor-like squashing: g(x)=1/(1+x^2)"""
    return 1.0 / (1.0 + x * x)


@dataclass
class LoopCandidate:
    iso: float
    poly3d: np.ndarray
    poly2d: np.ndarray
    face_ids: np.ndarray   # per-segment face ids
    seg_lens: np.ndarray   # per-segment lengths
    total_len: float
    base_score: float = 0.0
    vote_score: float = 0.0


def point_in_poly_2d(point: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    P = poly
    if P.shape[0] < 3:
        return False
    inside = False
    x0, y0 = P[-1, 0], P[-1, 1]
    for i in range(P.shape[0]):
        x1, y1 = P[i, 0], P[i, 1]
        if ((y1 > y) != (y0 > y)):
            xinters = (x0 - x1) * (y - y1) / ((y0 - y1) + 1e-12) + x1
            if x < xinters:
                inside = not inside
        x0, y0 = x1, y1
    return inside


def extract_loop_candidates(
    V: np.ndarray,
    F: np.ndarray,
    phi: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    lo: float,
    hi: float,
    n_isos: int = 120,
    stitch_tol: float = 1e-4,
    close_tol: float = 2e-3,
    min_points: int = 25,
) -> List[LoopCandidate]:
    right = np.asarray(right, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)

    def proj2d(P3):
        return np.c_[P3 @ right, P3 @ forward]

    isos = np.linspace(lo, hi, n_isos + 2)[1:-1]
    cands: List[LoopCandidate] = []

    for iso in isos:
        segs, fids = extract_isoline_segments(V, F, phi, float(iso))
        if segs.shape[0] == 0:
            continue

        comps = stitch_segments_to_polylines_with_ids(segs, fids, tol=stitch_tol)
        for poly3d, comp_face_ids, comp_seg_lens in comps:
            if poly3d.shape[0] < min_points:
                continue
            if not polyline_is_closed(poly3d, tol=close_tol):
                continue

            total_len = float(np.sum(comp_seg_lens))
            if total_len < 1e-10:
                continue

            cands.append(LoopCandidate(
                iso=float(iso),
                poly3d=poly3d,
                poly2d=proj2d(poly3d),
                face_ids=np.asarray(comp_face_ids, dtype=np.int32),
                seg_lens=np.asarray(comp_seg_lens, dtype=np.float64),
                total_len=total_len
            ))

    return cands


def compute_base_scores_dot_scissor(
    cands: List[LoopCandidate],
    tooth_peaks2d: np.ndarray,
    sigma_ms: float = 2.0,
    K_ms: int = 4,
    length_filter_ratio: float = 1.5,
) -> None:
    """
    base_score = Tightness * Concavity * Proximity
    - Tightness: l_min / l_i
    - Concavity: stepped concavity over ordered by iso
    - Proximity: mean distance from loop to tooth peaks (2D) squashed
    """
    if not cands:
        return

    # order by iso
    cands.sort(key=lambda c: c.iso)
    lengths = np.array([c.total_len for c in cands], dtype=np.float64)
    lmin = float(np.min(lengths) + 1e-12)

    # tightness
    tight = lmin / (lengths + 1e-12)
    tight = np.clip(tight, 0.0, 1.0)

    # concavity (multi-scale stepped concavity over lengths)
    conc = stepped_concavity_over_ordered(lengths, sigma=sigma_ms, K=K_ms)
    conc_norm = conc / (np.std(conc) + 1e-12)   # scale
    conc_term = g_squash(conc_norm)

    # proximity: loop close to tooth peaks (replace interactive click)
    prox = []
    for c in cands:
        # mean distance from a subsample of loop points to peaks2d
        P = c.poly2d
        if P.shape[0] > 800:
            P = P[:: max(1, P.shape[0] // 800)]
        # distance from each loop point to nearest peak
        # (small peak set, so brute force is fine)
        dmin = []
        for p in P:
            d = np.linalg.norm(tooth_peaks2d - p[None, :], axis=1)
            dmin.append(np.min(d))
        prox.append(np.mean(dmin))

    prox = np.asarray(prox, dtype=np.float64)
    prox = prox / (np.median(prox) + 1e-12)
    prox_term = g_squash(prox)

    # filter too-long candidates (Dot Scissor often filters long isolines)
    keep = lengths <= (length_filter_ratio * lmin)

    for i, c in enumerate(cands):
        if keep[i]:
            # also squash tight: prefer tight close to 1 (use inverse)
            tight_term = g_squash(1.0 / (tight[i] + 1e-12))
            c.base_score = float(conc_term[i] * tight_term * prox_term[i])
        else:
            c.base_score = 0.0


def face_based_voting_dot_scissor(
    cands: List[LoopCandidate],
    face_weight: np.ndarray,
) -> None:
    """
    Dot Scissor face-based voting:
      t_i = (Σ_j phi_j * l_ij) / (Σ_j l_ij)
      s_i = g_i * t_i
      Vote(j) = (Σ_i s_i * l_ij) / length(j)
    """
    if not cands:
        return

    n_faces = face_weight.shape[0]
    sum_len_face = np.zeros(n_faces, dtype=np.float64)
    sum_phi_len_face = np.zeros(n_faces, dtype=np.float64)

    for c in cands:
        if c.base_score <= 0.0:
            continue
        np.add.at(sum_len_face, c.face_ids, c.seg_lens)
        np.add.at(sum_phi_len_face, c.face_ids, c.seg_lens * c.base_score)

    t_i = sum_phi_len_face / (sum_len_face + 1e-12)
    s_i = face_weight * t_i

    for c in cands:
        if c.total_len <= 1e-10:
            c.vote_score = 0.0
            continue
        vote = float(np.sum(s_i[c.face_ids] * c.seg_lens))
        c.vote_score = vote / (c.total_len + 1e-12)


# ---------------------------
# Per-tooth selection
# ---------------------------

@dataclass
class ToothBoundaryDotScissor:
    tooth_index: int
    best: Optional[LoopCandidate]
    n_candidates: int
    coverage: float


# def pick_best_isoloops_per_tooth_dot_scissor(
#     all_cands: List[LoopCandidate],
#     teeth_peaks_indices: List[List[int]],
#     V: np.ndarray,
#     right: np.ndarray,
#     forward: np.ndarray,
#     face_weight: np.ndarray,
#     min_peak_ratio: float = 0.6,
#     sigma_ms: float = 2.0,
#     K_ms: int = 4,
#     length_filter_ratio: float = 1.5,
# ) -> List[ToothBoundaryDotScissor]:
#     """
#     For each tooth:
#       1) filter candidates that enclose enough peaks
#       2) compute base scores (tightness+concavity+proximity-to-peaks)
#       3) face-based voting within that tooth's candidate pool
#       4) pick max vote_score
#     """
#     right = np.asarray(right, dtype=np.float64)
#     forward = np.asarray(forward, dtype=np.float64)
#
#     def proj2d_pts(P3):
#         return np.c_[P3 @ right, P3 @ forward]
#
#     results: List[ToothBoundaryDotScissor] = []
#
#     for ti, peak_idx_list in enumerate(teeth_peaks_indices):
#         if len(peak_idx_list) == 0:
#             results.append(ToothBoundaryDotScissor(ti, None, 0, 0.0))
#             continue
#
#         peaks3d = V[np.asarray(peak_idx_list, dtype=np.int64)]
#         peaks2d = proj2d_pts(peaks3d)
#
#         # filter by containment (coverage)
#         tooth_cands: List[LoopCandidate] = []
#         best_cov = 0.0
#
#         for c in all_cands:
#             inside = 0
#             for puv in peaks2d:
#                 if point_in_poly_2d(puv, c.poly2d):
#                     inside += 1
#             cov = inside / max(1, len(peaks2d))
#             if cov >= min_peak_ratio:
#                 tooth_cands.append(c)
#                 if cov > best_cov:
#                     best_cov = cov
#
#         if not tooth_cands:
#             results.append(ToothBoundaryDotScissor(ti, None, 0, 0.0))
#             continue
#
#         # IMPORTANT: work on a shallow copy so per-tooth scores don't overwrite global
#         tooth_cands = [LoopCandidate(
#             iso=x.iso, poly3d=x.poly3d, poly2d=x.poly2d,
#             face_ids=x.face_ids, seg_lens=x.seg_lens, total_len=x.total_len
#         ) for x in tooth_cands]
#
#         # base score + voting
#         compute_base_scores_dot_scissor(
#             tooth_cands,
#             tooth_peaks2d=peaks2d,
#             sigma_ms=sigma_ms,
#             K_ms=K_ms,
#             length_filter_ratio=length_filter_ratio,
#         )
#         face_based_voting_dot_scissor(tooth_cands, face_weight)
#
#         best = max(tooth_cands, key=lambda z: z.vote_score)
#         results.append(ToothBoundaryDotScissor(
#             tooth_index=ti,
#             best=best,
#             n_candidates=len(tooth_cands),
#             coverage=best_cov
#         ))
#
#     return results
def pick_best_isoloops_per_tooth_dot_scissor(
    all_cands: List[LoopCandidate],
    teeth_peaks_indices: List[List[int]],
    V: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    face_weight: np.ndarray,
    min_peak_ratio: float = 0.6,
    sigma_ms: float = 2.0,
    K_ms: int = 4,
    length_filter_ratio: float = 1.5,
    # --- new: exclusivity only ---
    max_other_peak_ratio: float = 0.2,  # 0.1~0.3 建议
) -> List[ToothBoundaryDotScissor]:
    """
    For each tooth:
      1) filter candidates that enclose enough peaks (coverage)
      2) exclusivity: reject loops that also enclose other teeth peaks too much
      3) compute base scores (tightness+concavity+proximity-to-peaks)
      4) face-based voting within that tooth's candidate pool
      5) pick max vote_score
    """
    right = np.asarray(right, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)

    def proj2d_pts(P3):
        return np.c_[P3 @ right, P3 @ forward]

    results: List[ToothBoundaryDotScissor] = []

    # ---- cache all teeth peaks in 2D once (for exclusivity checking) ----
    all_teeth_peaks2d: List[Optional[np.ndarray]] = []
    for peak_idx_list in teeth_peaks_indices:
        if len(peak_idx_list) == 0:
            all_teeth_peaks2d.append(None)
        else:
            peaks3d = V[np.asarray(peak_idx_list, dtype=np.int64)]
            all_teeth_peaks2d.append(proj2d_pts(peaks3d))

    for ti, peak_idx_list in enumerate(teeth_peaks_indices):
        if len(peak_idx_list) == 0:
            results.append(ToothBoundaryDotScissor(ti, None, 0, 0.0))
            continue

        peaks2d = all_teeth_peaks2d[ti]
        assert peaks2d is not None

        # filter by containment (coverage) + exclusivity
        tooth_cands: List[LoopCandidate] = []
        best_cov = 0.0

        for c in all_cands:
            # (1) coverage for current tooth
            inside = 0
            for puv in peaks2d:
                if point_in_poly_2d(puv, c.poly2d):
                    inside += 1
            cov = inside / max(1, len(peaks2d))
            if cov < min_peak_ratio:
                continue

            # (2) exclusivity: reject if it also covers other teeth too much
            other_max = 0.0
            for tj, other_peaks2d in enumerate(all_teeth_peaks2d):
                if tj == ti or other_peaks2d is None:
                    continue
                inside_o = 0
                for puv in other_peaks2d:
                    if point_in_poly_2d(puv, c.poly2d):
                        inside_o += 1
                cov_o = inside_o / max(1, len(other_peaks2d))
                if cov_o > other_max:
                    other_max = cov_o
                if other_max > max_other_peak_ratio:
                    break

            if other_max > max_other_peak_ratio:
                continue

            tooth_cands.append(c)
            if cov > best_cov:
                best_cov = cov

        if not tooth_cands:
            results.append(ToothBoundaryDotScissor(ti, None, 0, 0.0))
            continue

        # IMPORTANT: work on a shallow copy so per-tooth scores don't overwrite global
        tooth_cands = [LoopCandidate(
            iso=x.iso, poly3d=x.poly3d, poly2d=x.poly2d,
            face_ids=x.face_ids, seg_lens=x.seg_lens, total_len=x.total_len
        ) for x in tooth_cands]

        # base score + voting (unchanged)
        compute_base_scores_dot_scissor(
            tooth_cands,
            tooth_peaks2d=peaks2d,
            sigma_ms=sigma_ms,
            K_ms=K_ms,
            length_filter_ratio=length_filter_ratio,
        )
        face_based_voting_dot_scissor(tooth_cands, face_weight)

        best = max(tooth_cands, key=lambda z: z.vote_score)
        results.append(ToothBoundaryDotScissor(
            tooth_index=ti,
            best=best,
            n_candidates=len(tooth_cands),
            coverage=best_cov
        ))

    return results
