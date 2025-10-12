""" some util functions for trimesh"""

from collections import defaultdict
from trimesh import Trimesh
import numpy as np


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
        curvature_map = np.array([d[i] for i in range(len(mesh.faces))])
        return curvature, curvature_map

    return curvature

def get_local_maximum_along_dir(mesh, direction) -> np.ndarray :
    # create the vertex mask to be returned
    # init one to all vertex
    mask = np.ones(len(mesh.vertices), dtype=bool)

    # calculate height for all vertex
    #direction_mat = direction.reshape(-1, 1)
    heights = np.dot(mesh.vertices, direction)
    # get idx (= 0,1,2) with the maximum value in each face
    heights_per_face = heights[mesh.faces]
    max_idx = np.argmax(heights_per_face, axis=1)
    # get non-max vertex index
    non_max_mask = np.ones_like(mesh.faces, dtype=bool)
    non_max_mask[np.arange(mesh.faces.shape[0]), max_idx] = 0
    non_max_idx = mesh.faces[non_max_mask]

    mask[non_max_idx] = 0
    mask = np.nonzero(mask)[0]
    return mask