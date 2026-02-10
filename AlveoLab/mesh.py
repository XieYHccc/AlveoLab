import trimesh as tm
import numpy as np

from AlveoLab.utils import LazyAttribute
from AlveoLab.trimesh_utils import get_edge_based_curvature, get_face_face_adjacency

class Mesh:
    def __init__(self, trimesh_obj: tm.Trimesh):
        if not isinstance(trimesh_obj, tm.Trimesh):
            raise TypeError(f"Mesh expects a trimesh.Trimesh, got {type(trimesh_obj)!r}")
        self._mesh = trimesh_obj

    # ---- alternative constructors ----
    @classmethod
    def from_file(cls, file_path):
        """
        Construct from a mesh file path.
        Any load_kwargs are passed to trimesh.load_mesh.
        """
        return cls(tm.load_mesh(file_path))

    @classmethod
    def from_vertices_faces(cls, vertices: np.ndarray, faces: np.ndarray):
        m = tm.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces))
        return cls(m)

    def __getattr__(self, name):
        return getattr(self._mesh, name)

    @LazyAttribute
    def face_neighbors(self):
        # non-ordered
        return get_face_face_adjacency(self._mesh)

    @LazyAttribute
    def tri2tri_edge_curvatures(self):
        _, tri2tri_edge_curvature = get_edge_based_curvature(self._mesh, True)
        return tri2tri_edge_curvature

    @LazyAttribute
    def tri2tri_displacements(self):
        """
        Displacement from each face center to each neighbour face center.

        Returns
        -------
        (n_faces, 3, 3) float array.
        For missing neighbours (index == -1), displacement is nan.
        NOTE: neighbour order is not tied to specific edges (non-ordered).
        """
        centers = np.asarray(self._mesh.triangles_center)   # (n_faces, 3)
        neigh = np.asarray(self.face_neighbors)            # (n_faces, 3)

        disp = np.full((len(centers), 3, 3), np.nan, dtype=centers.dtype)
        valid = neigh != -1
        fi, k = np.nonzero(valid)
        disp[fi, k] = centers[neigh[fi, k]] - centers[fi]
        return disp


