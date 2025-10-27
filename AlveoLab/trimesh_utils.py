""" some util functions for trimesh"""
from collections import defaultdict

from trimesh import Trimesh
from trimesh.bounds import oriented_bounds
import numpy as np
import networkx as nx
import scipy.sparse as sp

from AlveoLab.math.oriented_bounding_box import Obb
from AlveoLab.math.geometry import cotangent


def get_face_face_adjacency(mesh: Trimesh):
    """
    get each face's three adjacent faces

    Parameters
    -------------
    mesh: Trimesh object

    Returns
    ----------
    face_neighbors: (len(mesh.faces), 3)int
        Indexes of 3 adjacent faces.
        For non-watertight mesh, the boundary faces will have -1 as the adjacent face index.
    """
    face_adj = mesh.face_adjacency

    # create a default dict object to store the neighbors of each face the default value of dictionary is empty list.
    d = defaultdict(list)
    [(d[a].append(b), d[b].append(a)) for a, b in mesh.face_adjacency]

    face_neighbors = np.array([d[i] + [-1] * (3 - len(d[i])) for i in range(len(mesh.faces))])

    return face_neighbors


def get_edge_based_curvature(mesh: Trimesh, get_map=False):
    """
        correspond to trimesh.Trimesh.face_adjacency, ignore edges on the boundary,
        but for watertight mesh, face_adjacency also correspond to edges_unique.
        edge based curvature = |(n0 x n1)|/|△x|.Triangles are referenced by
        argument based on the order they are listed in Trimesh.faces
    """

    face_adj = mesh.face_adjacency
    face_normals = mesh.face_normals

    # get face normals for each pair
    face_normals_adj = face_normals[face_adj]
    cross_product = np.cross(face_normals_adj[:, 0, :], face_normals_adj[:, 1, :])
    curvature = np.linalg.norm(cross_product, axis=1)
    face_center_pair = mesh.triangles_center[face_adj]

    # calculate vector form face0's center to face1's center
    c2c_vector = np.diff(face_center_pair, axis=1).reshape(-1, 3)
    c2c_vector_norm = np.linalg.norm(c2c_vector, axis=1)

    curvature = curvature / c2c_vector_norm
    sign = np.sign(np.sum(face_normals_adj[:, 0, :] * c2c_vector, axis=1))
    curvature = -sign * curvature

    if get_map:
        d = defaultdict(list)
        [(d[a].append(curvature[i]), d[b].append(curvature[i])) for i, (a, b) in enumerate(mesh.face_adjacency)]
        curvature_map = np.array([
            d[i] if len(d[i]) == 3 else [np.nan, np.nan, np.nan]
            for i in range(len(mesh.faces))
        ])
        return curvature, curvature_map

    return curvature


def get_local_maximum_along_dir(mesh, direction) -> np.ndarray:
    heights = np.dot(mesh.vertices, direction)
    return get_local_maximum(mesh, heights)


def get_local_maximum(mesh, scalar_filed):
    # init one to all vertex
    mask = np.ones(len(mesh.vertices), dtype=bool)

    scalars_per_triangle = scalar_filed[mesh.faces]
    max_idx = np.argmax(scalars_per_triangle, axis=1)
    # get non-max vertex index
    non_max_mask = np.ones_like(mesh.faces, dtype=bool)
    non_max_mask[np.arange(mesh.faces.shape[0]), max_idx] = 0
    non_max_idx = mesh.faces[non_max_mask]

    mask[non_max_idx] = 0
    indices = np.nonzero(mask)[0]
    return indices


def get_oriented_bounding_box(mesh):
    to_origin, extents = oriented_bounds(mesh)
    obb = Obb(to_origin, extents)

    return obb


def discrete_mean_curvature_measure(mesh):
    """Calculate discrete mean curvature of mesh using one-ring neighborhood."""

    # one-rings (immediate neighbors of) each vertex
    g = nx.from_edgelist(mesh.edges_unique)
    one_rings = [list(g[i].keys()) for i in range(len(mesh.vertices))]

    # cotangents of angles and store in dictionary based on corresponding vertex and face
    face_angles = mesh.face_angles_sparse
    cotangents = {f"{vertex},{face}": 1 / np.tan(angle) for vertex, face, angle in
                  zip(face_angles.row, face_angles.col, face_angles.data)}

    # discrete Laplace-Beltrami contribution of the shared edge of adjacent faces:
    #        /*\
    #       / * \
    #      /  *  \
    #    vi___*___vj
    #
    # store results in dictionary with vertex ids as keys
    fa = mesh.face_adjacency
    fae = mesh.face_adjacency_edges
    edge_measure = {f"{fae[i][0]},{fae[i][1]}": (mesh.vertices[fae[i][1]] - mesh.vertices[fae[i][0]]) * (
            cotangents[f"{v[0]},{fa[i][0]}"] + cotangents[f"{v[1]},{fa[i][1]}"]) for i, v in
                    enumerate(mesh.face_adjacency_unshared)}

    # calculate mean curvature using one-ring
    mean_curv = [0] * len(mesh.vertices)
    for vertex_id, face_ids in enumerate(mesh.vertex_faces):
        face_ids = face_ids[face_ids != -1]  # faces associated with vertex_id
        one_ring = one_rings[vertex_id]
        # delta_s = 0
        delta_s = np.zeros(3)

        for one_ring_vertex_id in one_ring:
            if f"{vertex_id},{one_ring_vertex_id}" in edge_measure:
                delta_s += edge_measure[f"{vertex_id},{one_ring_vertex_id}"]
            elif f"{one_ring_vertex_id},{vertex_id}" in edge_measure:
                delta_s -= edge_measure[f"{one_ring_vertex_id},{vertex_id}"]

        delta_s *= 1 / (2 * sum(mesh.area_faces[face_ids]) / 3)  # use 1/3 of the areas
        n = mesh.vertex_normals[vertex_id]
        # sign = np.sign(-np.dot(n, delta_s))
        # mean_curv[vertex_id] = 0.5 * np.linalg.norm(delta_s) * sign
        mean_curv[vertex_id] = -0.5 * np.dot(n, delta_s)

    return np.array(mean_curv)


def smooth_curvature(mesh, curvature, iterations=3):
    """Simple neighborhood averaging of vertex curvature."""
    g = nx.from_edgelist(mesh.edges_unique)
    smoothed = curvature.copy().astype(float)

    for _ in range(iterations):
        new_curv = smoothed.copy()
        for vid, nbrs in enumerate(g):
            nbrs = list(g[vid].keys())
            if not nbrs:
                continue
            new_curv[vid] = np.mean(smoothed[nbrs + [vid]])
        smoothed = new_curv
    return smoothed


def get_cotangent_weights_laplacian_matrix(mesh):
    """
    Compute cotangent weights for a triangular mesh
    Inputs:
        V: (n,3) array of vertex positions
        F: (m,3) array of triangle indices
    Returns:
        W: sparse (n,n) symmetric matrix of cotangent weights
    """

    vertices = mesh.vertices
    faces = mesh.faces
    n = len(vertices)
    I, J, W = [], [], []

    for tri in faces:
        i, j, k = tri
        vi, vj, vk = vertices[i], vertices[j], vertices[k]

        # 计算三个角的 cot 值
        cot_alpha = cotangent(vj, vi, vk)
        cot_beta = cotangent(vi, vj, vk)
        cot_gamma = cotangent(vi, vk, vj)

        # 累积权重 (对称)
        for (p, q, w) in [(i, j, cot_gamma), (j, k, cot_alpha), (k, i, cot_beta)]:
            I += [p, q]
            J += [q, p]
            W += [w, w]

    # 构造稀疏矩阵
    W = sp.coo_matrix((W, (I, J)), shape=(n, n))
    # 对每条边权重取 1/2 (标准定义)
    W *= 0.5

    d = np.asarray(W.sum(axis=1)).ravel()
    D = sp.diags(d)

    return D - W


def build_Ab_from_L_and_constraints(L, n, fs_idx, bs_idx, us_idx, w=1000.0):
    """
    L: (n,n) sparse Laplacian
    n: number of vertices
    fs_idx, bs_idx, us_idx: lists/arrays of vertex indices (FS/BS/US)
    w: constraint weight (large)
    Returns:
        A (sparse), b (dense)
    """
    # 构造 C
    m = len(fs_idx) + len(bs_idx) + len(us_idx)
    rows = []
    cols = []
    data = []
    # FS rows
    r = 0
    for i in fs_idx:
        rows.append(r);
        cols.append(i);
        data.append(w);
        r += 1
    # BS rows
    for i in bs_idx:
        rows.append(r);
        cols.append(i);
        data.append(w);
        r += 1
    # US rows
    for i in us_idx:
        rows.append(r);
        cols.append(i);
        data.append(w);
        r += 1

    C = sp.coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()

    # 构造 b0（对应 C 的目标值）
    b0 = np.zeros(m, dtype=float)
    # FS -> 1, BS -> 0, US -> 0.5（都乘 w 已体现在 C 的行系数）
    if len(fs_idx) > 0:
        b0[:len(fs_idx)] = w * 1.0
    if len(us_idx) > 0:
        b0[len(fs_idx) + len(bs_idx):] = w * 0.5

    # A = [L; C], b = [0; b0]
    A = sp.vstack([L.tocsr(), C], format='csr')
    b = np.concatenate([np.zeros(n, dtype=float), b0])
    return A, b
