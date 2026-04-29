from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from AlveoLab.mhb.frame import GlobalFrame
from AlveoLab.mhb.labels import ToothLabelDefinition


@dataclass
class MhbKeypoint:
    kind: str
    point: np.ndarray
    vertex_index: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToothKeypointResult:
    label: int
    tooth_name: str
    tooth_family: str
    recognizer_name: str
    keypoints: list[MhbKeypoint]
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToothContext:
    mesh: Any
    label: int
    definition: ToothLabelDefinition
    vertex_labels: np.ndarray
    vertex_indices: np.ndarray
    face_indices: np.ndarray
    face_mask: np.ndarray
    points: np.ndarray
    center: np.ndarray
    global_frame: GlobalFrame
    mesiodistal_axis: np.ndarray
    buccolingual_axis: np.ndarray
    occlusal_axis: np.ndarray
    mesiodistal_values: np.ndarray
    buccolingual_values: np.ndarray
    occlusal_values: np.ndarray
    boundary_mask: np.ndarray
    horizontal_scale: float
    axis_source: str
