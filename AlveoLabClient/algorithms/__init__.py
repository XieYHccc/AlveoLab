"""Algorithm strategies and registry for the AlveoLab client.

Extension recipe
----------------
To add a new segmentation method:

1. Implement a class with ``name: str`` and
   ``run(mesh, arch_type) -> SegmentationResult``.
2. Register it in :func:`default_registry` (or inject your own
   :class:`AlgorithmRegistry` into the main window).
3. Restart the app — the Tools menu picks it up automatically. With multiple
   methods registered, the menu auto-promotes the category to a submenu.

Landmark and assessment categories follow the same pattern. The UI shell does
not need to change.
"""

from .base import (
    AssessmentAlgorithm,
    Landmark,
    LandmarkAlgorithm,
    LandmarkResult,
    SegmentationAlgorithm,
    SegmentationResult,
)
from .landmark_recognizer_backed import (
    LandmarkRecognizerLandmarks,
    LandmarkRecognizerSegmentation,
)
from .registry import AlgorithmRegistry, default_registry

__all__ = [
    "AlgorithmRegistry",
    "AssessmentAlgorithm",
    "Landmark",
    "LandmarkAlgorithm",
    "LandmarkRecognizerLandmarks",
    "LandmarkRecognizerSegmentation",
    "LandmarkResult",
    "SegmentationAlgorithm",
    "SegmentationResult",
    "default_registry",
]
