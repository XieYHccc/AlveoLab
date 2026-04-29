from __future__ import annotations

import numpy as np

from AlveoLab.math.geometry import normalize_vector
from AlveoLab.mhb.frame import GlobalFrame
from AlveoLab.mesh import Mesh


def as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def scale_to_unit_interval(values: np.ndarray) -> np.ndarray:
    values = as_float_array(values)
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-8:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def clip_percentile_range(
    values: np.ndarray,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
) -> tuple[np.ndarray, float, float]:
    values = as_float_array(values)
    if values.size == 0:
        return values, 0.0, 0.0

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.zeros_like(values), 0.0, 0.0

    finite_values = values[finite_mask]
    lower = float(np.percentile(finite_values, lower_percentile))
    upper = float(np.percentile(finite_values, upper_percentile))

    clipped = values.copy()
    clipped[finite_mask] = np.clip(finite_values, lower, upper)
    clipped[~finite_mask] = lower
    return clipped, lower, upper


def ensure_mesh(mesh) -> Mesh:
    if isinstance(mesh, Mesh):
        return mesh
    return Mesh(mesh)


def largest_connected_component(mesh, vertex_indices: np.ndarray) -> np.ndarray:
    if vertex_indices.size <= 1:
        return vertex_indices

    allowed = set(int(i) for i in vertex_indices.tolist())
    remaining = set(allowed)
    best_component: list[int] = []

    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            for neighbor in mesh.vertex_neighbors[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)

        if len(component) > len(best_component):
            best_component = component

    return np.array(sorted(best_component), dtype=np.int64)


def estimate_tooth_axes_from_pca(points: np.ndarray, frame: GlobalFrame) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0)
    flattened = centered - np.outer(centered @ frame.occlusal, frame.occlusal)
    if np.allclose(flattened, 0.0):
        mesiodistal = normalize_vector(frame.right)
    else:
        _, _, vh = np.linalg.svd(flattened, full_matrices=False)
        mesiodistal = vh[0]
        mesiodistal = mesiodistal - np.dot(mesiodistal, frame.occlusal) * frame.occlusal
        if np.linalg.norm(mesiodistal) < 1e-8:
            mesiodistal = frame.right
        mesiodistal = normalize_vector(mesiodistal)

    if np.dot(mesiodistal, frame.right) < 0:
        mesiodistal = -mesiodistal

    buccolingual = normalize_vector(np.cross(frame.occlusal, mesiodistal))
    if np.dot(buccolingual, frame.forward) < 0:
        buccolingual = -buccolingual
    mesiodistal = normalize_vector(np.cross(buccolingual, frame.occlusal))
    return mesiodistal, buccolingual


def build_boundary_mask(mesh, vertex_indices: np.ndarray, vertex_labels: np.ndarray, label: int) -> np.ndarray:
    boundary_mask = np.zeros(vertex_indices.shape[0], dtype=bool)
    for local_index, vertex_index in enumerate(vertex_indices):
        for neighbor in mesh.vertex_neighbors[int(vertex_index)]:
            if int(vertex_labels[neighbor]) != label:
                boundary_mask[local_index] = True
                break
    return boundary_mask
