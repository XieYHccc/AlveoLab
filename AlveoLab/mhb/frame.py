from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from AlveoLab.math.geometry import normalize_vector
from AlveoLab.mhb.orienter import ToothRegionPcaOrienter
from AlveoLab.orienter.pca_dental_orienter import PcaOrienter


@dataclass(frozen=True)
class GlobalFrame:
    right: np.ndarray
    forward: np.ndarray
    occlusal: np.ndarray
    center: np.ndarray

    @classmethod
    def from_mesh(
        cls,
        mesh,
        arch_type: str | None = None,
        occlusal_axis: np.ndarray | None = None,
        vertex_labels: np.ndarray | None = None,
        orienter=None,
    ):
        if orienter is None:
            if vertex_labels is not None:
                orienter = ToothRegionPcaOrienter(mesh, vertex_labels, arch_type or "L")
            else:
                orienter = PcaOrienter(mesh, arch_type or "L")

        right = np.asarray(orienter.right, dtype=float)
        forward = np.asarray(orienter.forward, dtype=float)
        occlusal = np.asarray(orienter.occlusal, dtype=float)

        if occlusal_axis is not None:
            occlusal = normalize_vector(np.asarray(occlusal_axis, dtype=float))
            right = right - np.dot(right, occlusal) * occlusal
            if np.linalg.norm(right) < 1e-8:
                right = forward - np.dot(forward, occlusal) * occlusal
            right = normalize_vector(right)
            forward = normalize_vector(np.cross(occlusal, right))
            right = normalize_vector(np.cross(forward, occlusal))

        return cls(
            right=right,
            forward=forward,
            occlusal=occlusal,
            center=np.asarray(orienter.center, dtype=float),
        )
