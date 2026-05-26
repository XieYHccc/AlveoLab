"""Segmentation + landmark strategies that delegate to `LandmarkRecognizer`.

`AlveoLab.landmark_recognizer.LandmarkRecognizer.__init__` runs the full
pipeline (orientation → peak filtering → `CurvatureBasedSeg` → label arrays),
so both strategies just construct it and pluck the relevant attributes.

v1 deliberately does not share a `LandmarkRecognizer` instance between the two
strategies — each click recomputes. A scene-level cache can be added later
without touching the strategy interface.
"""

from __future__ import annotations

import numpy as np

from AlveoLab.landmark_recognizer import LandmarkRecognizer

from .base import Landmark, LandmarkResult, SegmentationResult


class LandmarkRecognizerSegmentation:
    """Run `LandmarkRecognizer` and return its per-vertex/per-face tooth labels."""

    name = "LandmarkRecognizer (curvature)"

    def run(self, mesh, arch_type: str) -> SegmentationResult:
        recognizer = LandmarkRecognizer(mesh, arch_type)
        vertex_labels = np.asarray(recognizer.teeth_vertex_labels, dtype=np.int64)
        face_labels = np.asarray(recognizer.teeth_face_labels, dtype=np.int64)
        # `LandmarkRecognizer._preprocess_mesh` crops the gum off, so the label
        # arrays describe `recognizer.mesh`, not the original input. Pass that
        # cropped mesh through so the viewport can swap to it.
        return SegmentationResult(
            vertex_labels=vertex_labels,
            face_labels=face_labels,
            mesh=recognizer.mesh,
            metadata={
                "source": "LandmarkRecognizer",
                "num_teeth_found": len(recognizer.teeth),
            },
        )


class LandmarkRecognizerLandmarks:
    """Run `LandmarkRecognizer` and expose its filtered peaks as landmarks."""

    name = "LandmarkRecognizer (peaks)"

    def run(self, mesh, arch_type: str) -> LandmarkResult:
        recognizer = LandmarkRecognizer(mesh, arch_type)
        landmarks: list[Landmark] = []
        for peak in recognizer.peaks:
            landmarks.append(
                Landmark(
                    point=np.asarray(peak.point, dtype=float),
                    kind="peak",
                    vertex_index=int(peak.index),
                )
            )
        return LandmarkResult(
            landmarks=landmarks,
            metadata={
                "source": "LandmarkRecognizer",
                "num_peaks": len(landmarks),
            },
        )
