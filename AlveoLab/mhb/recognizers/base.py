from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from AlveoLab.mhb.models import MhbKeypoint, ToothContext, ToothKeypointResult


class BaseToothKeypointRecognizer(ABC):
    name = "base"

    @abstractmethod
    def recognize(self, context: ToothContext) -> ToothKeypointResult:
        raise NotImplementedError

    def _make_keypoint(
        self,
        context: ToothContext,
        local_index: int,
        kind: str,
        score: float,
        **metadata,
    ) -> MhbKeypoint:
        return MhbKeypoint(
            kind=kind,
            point=np.asarray(context.points[local_index], dtype=float),
            vertex_index=int(context.vertex_indices[local_index]),
            score=float(score),
            metadata=metadata,
        )

    def _result(
        self,
        context: ToothContext,
        keypoints: list[MhbKeypoint],
        **debug,
    ) -> ToothKeypointResult:
        return ToothKeypointResult(
            label=context.label,
            tooth_name=context.definition.name,
            tooth_family=context.definition.family,
            recognizer_name=self.name,
            keypoints=keypoints,
            debug=debug,
        )

