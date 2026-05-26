"""Strategy interfaces for the client's pluggable algorithms.

A v1 prototype exposes two categories: segmentation and landmark detection.
Assessment is reserved as a placeholder for v2. To add a new method, implement
the relevant `Protocol`, register an instance in `default_registry()`, and the
Tools menu picks it up on next launch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


# ---- result types ---------------------------------------------------------


@dataclass
class SegmentationResult:
    """Output of any segmentation algorithm.

    `vertex_labels` is required (shape (n_vertices,), int, 0=gum, 1..N=teeth).
    `face_labels` is optional but recommended — `MeshView` prefers face labels
    because cell scalars give cleaner per-tooth coloring than vertex scalars.

    `mesh` is optional. Some algorithms (e.g. `LandmarkRecognizer`) crop the
    mesh internally (gum removal) before producing labels, so the labels do
    not line up with the original input mesh. When the algorithm sets `mesh`,
    its `vertex_labels` / `face_labels` refer to *that* mesh; the viewport
    will swap to it before applying the coloring. When `mesh` is None the
    labels are assumed to match the input mesh.
    """

    vertex_labels: np.ndarray
    face_labels: np.ndarray | None = None
    mesh: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_teeth(self) -> int:
        labels = self.vertex_labels
        if labels.size == 0:
            return 0
        return int(np.max(labels[labels > 0])) if (labels > 0).any() else 0


@dataclass
class Landmark:
    point: np.ndarray  # (3,) float in world coordinates
    kind: str = "peak"
    score: float = 0.0
    vertex_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LandmarkResult:
    landmarks: list[Landmark]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---- strategy protocols ---------------------------------------------------


@runtime_checkable
class SegmentationAlgorithm(Protocol):
    """A segmentation strategy. `name` is shown in the Tools menu."""

    name: str

    def run(self, mesh: Any, arch_type: str) -> SegmentationResult: ...


@runtime_checkable
class LandmarkAlgorithm(Protocol):
    """A landmark-detection strategy. `name` is shown in the Tools menu."""

    name: str

    def run(self, mesh: Any, arch_type: str) -> LandmarkResult: ...


@runtime_checkable
class AssessmentAlgorithm(Protocol):
    """Reserved for v2. Currently unused — present so the registry has a slot."""

    name: str

    def run(
        self,
        mesh: Any,
        arch_type: str,
        segmentation: SegmentationResult | None,
        landmarks: LandmarkResult | None,
    ) -> Any: ...
