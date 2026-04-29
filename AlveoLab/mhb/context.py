from __future__ import annotations

import numpy as np

from AlveoLab.mhb.frame import GlobalFrame
from AlveoLab.mhb.labels import TOOTH_LABEL_DEFINITIONS
from AlveoLab.mhb.medial_curve import ArchMedialCurve
from AlveoLab.mhb.models import ToothContext
from AlveoLab.mhb.utils import build_boundary_mask, ensure_mesh, estimate_tooth_axes_from_pca, largest_connected_component


def build_tooth_context(
    mesh,
    vertex_labels: np.ndarray,
    label: int,
    frame: GlobalFrame,
    medial_curve: ArchMedialCurve | None = None,
    keep_largest_component: bool = True,
) -> ToothContext | None:
    mesh = ensure_mesh(mesh)
    definition = TOOTH_LABEL_DEFINITIONS.get(int(label))
    if definition is None:
        return None

    vertex_indices = np.flatnonzero(vertex_labels == int(label)).astype(np.int64)
    if vertex_indices.size == 0:
        return None

    if keep_largest_component:
        vertex_indices = largest_connected_component(mesh, vertex_indices)

    selected_vertex_mask = np.zeros(np.asarray(mesh.vertices).shape[0], dtype=bool)
    selected_vertex_mask[vertex_indices] = True
    face_membership = selected_vertex_mask[np.asarray(mesh.faces)].sum(axis=1)
    face_mask = face_membership >= 2
    face_indices = np.flatnonzero(face_mask).astype(np.int64)

    points = np.asarray(mesh.vertices[vertex_indices], dtype=float)
    center = points.mean(axis=0)
    axis_source = "pca_fallback"
    if medial_curve is not None:
        try:
            mesiodistal_axis, buccolingual_axis = medial_curve.estimate_tooth_axes(points)
            axis_source = "medial_curve"
        except Exception:
            mesiodistal_axis, buccolingual_axis = estimate_tooth_axes_from_pca(points, frame)
    else:
        mesiodistal_axis, buccolingual_axis = estimate_tooth_axes_from_pca(points, frame)
    occlusal_axis = frame.occlusal

    mesiodistal_values = points @ mesiodistal_axis
    buccolingual_values = points @ buccolingual_axis
    occlusal_values = points @ occlusal_axis
    boundary_mask = build_boundary_mask(mesh, vertex_indices, vertex_labels, int(label))

    horizontal_scale = max(
        float(np.ptp(mesiodistal_values)) if mesiodistal_values.size else 0.0,
        float(np.ptp(buccolingual_values)) if buccolingual_values.size else 0.0,
        1e-6,
    )

    return ToothContext(
        mesh=mesh,
        label=int(label),
        definition=definition,
        vertex_labels=vertex_labels,
        vertex_indices=vertex_indices,
        face_indices=face_indices,
        face_mask=face_mask,
        points=points,
        center=center,
        global_frame=frame,
        mesiodistal_axis=mesiodistal_axis,
        buccolingual_axis=buccolingual_axis,
        occlusal_axis=occlusal_axis,
        mesiodistal_values=mesiodistal_values,
        buccolingual_values=buccolingual_values,
        occlusal_values=occlusal_values,
        boundary_mask=boundary_mask,
        horizontal_scale=horizontal_scale,
        axis_source=axis_source,
    )
