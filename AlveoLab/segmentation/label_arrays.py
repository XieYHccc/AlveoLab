import numpy as np


def build_face_label_array(mesh, face_masks, tooth_keys) -> np.ndarray:
    face_labels = np.zeros(mesh.faces.shape[0], dtype=np.int64)
    for face_mask, key in zip(face_masks, tooth_keys):
        if face_mask is None:
            continue
        face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
        if face_mask.shape[0] != face_labels.shape[0]:
            raise ValueError("face_mask length does not match the mesh face count.")
        face_labels[face_mask] = int(key)
    return face_labels


def build_vertex_label_array(mesh, face_masks, tooth_keys) -> np.ndarray:
    vertex_labels = np.zeros(mesh.vertices.shape[0], dtype=np.int64)
    for face_mask, key in zip(face_masks, tooth_keys):
        if face_mask is None:
            continue
        face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
        if face_mask.shape[0] != mesh.faces.shape[0]:
            raise ValueError("face_mask length does not match the mesh face count.")
        if not np.any(face_mask):
            continue
        vertex_ids = np.unique(mesh.faces[face_mask].reshape(-1))
        vertex_labels[vertex_ids] = int(key)
    return vertex_labels


def build_face_and_vertex_label_arrays(mesh, face_masks, tooth_keys):
    face_labels = build_face_label_array(mesh, face_masks, tooth_keys)
    vertex_labels = build_vertex_label_array(mesh, face_masks, tooth_keys)
    return face_labels, vertex_labels
