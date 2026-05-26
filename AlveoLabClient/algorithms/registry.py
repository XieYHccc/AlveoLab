"""Registry of available algorithms, keyed by display name.

The `MainWindow` reads this registry to build the Tools menu. The shape of the
menu follows the registry shape automatically:

* a category with one entry renders as a single menu item,
* a category with multiple entries renders as a submenu listing each name.

No UI code changes are needed when a new algorithm is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import AssessmentAlgorithm, LandmarkAlgorithm, SegmentationAlgorithm
from .landmark_recognizer_backed import (
    LandmarkRecognizerLandmarks,
    LandmarkRecognizerSegmentation,
)


@dataclass
class AlgorithmRegistry:
    segmentation: dict[str, SegmentationAlgorithm] = field(default_factory=dict)
    landmark: dict[str, LandmarkAlgorithm] = field(default_factory=dict)
    assessment: dict[str, AssessmentAlgorithm] = field(default_factory=dict)


def default_registry() -> AlgorithmRegistry:
    """Build the registry shipped with the v1 prototype."""
    reg = AlgorithmRegistry()

    seg = LandmarkRecognizerSegmentation()
    reg.segmentation[seg.name] = seg

    lm = LandmarkRecognizerLandmarks()
    reg.landmark[lm.name] = lm

    # reg.assessment[...] - placeholder for v2

    return reg
