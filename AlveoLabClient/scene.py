"""Document-level state shared between the menus and the viewport.

`SceneModel` is a `QObject` so widgets can subscribe to fine-grained signals.
It is the single source of truth: `MainWindow` mutates it, `MeshView` listens.

Signals are intentionally narrow: each one corresponds to one layer of the
viewport (base mesh / segmentation coloring / landmark spheres). A consumer
that only cares about one layer does not have to re-read the whole state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from .algorithms import LandmarkResult, SegmentationResult


class SceneModel(QObject):
    mesh_loaded = Signal(object)            # (mesh,) — emitted after a new mesh is set
    segmentation_updated = Signal(object)   # (SegmentationResult,)
    landmarks_updated = Signal(object)      # (LandmarkResult,)
    cleared = Signal()                      # emitted when the scene is reset

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._mesh: Any = None
        self._arch_type: str | None = None
        self._source_path: Path | None = None
        self._segmentation: SegmentationResult | None = None
        self._landmarks: LandmarkResult | None = None

    # ---- read-only accessors ----------------------------------------------

    @property
    def mesh(self) -> Any:
        return self._mesh

    @property
    def arch_type(self) -> str | None:
        return self._arch_type

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def segmentation(self) -> SegmentationResult | None:
        return self._segmentation

    @property
    def landmarks(self) -> LandmarkResult | None:
        return self._landmarks

    @property
    def has_mesh(self) -> bool:
        return self._mesh is not None

    # ---- mutators ---------------------------------------------------------

    def set_mesh(self, mesh: Any, arch_type: str, source_path: Path | None = None) -> None:
        """Install a new mesh; clears any previous segmentation/landmarks."""
        self._mesh = mesh
        self._arch_type = arch_type
        self._source_path = Path(source_path) if source_path is not None else None
        self._segmentation = None
        self._landmarks = None
        self.mesh_loaded.emit(mesh)

    def set_segmentation(self, result: SegmentationResult) -> None:
        self._segmentation = result
        self.segmentation_updated.emit(result)

    def set_landmarks(self, result: LandmarkResult) -> None:
        self._landmarks = result
        self.landmarks_updated.emit(result)

    def clear(self) -> None:
        self._mesh = None
        self._arch_type = None
        self._source_path = None
        self._segmentation = None
        self._landmarks = None
        self.cleared.emit()
