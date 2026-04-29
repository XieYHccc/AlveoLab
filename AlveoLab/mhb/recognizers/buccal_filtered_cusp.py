from __future__ import annotations

from dataclasses import replace

import numpy as np

from AlveoLab.mhb.models import ToothContext, ToothKeypointResult
from AlveoLab.mhb.recognizers.cusp_based import CuspBasedMhbKeypointRecognizer


class BuccalFilteredCuspRecognizer(CuspBasedMhbKeypointRecognizer):
    """
    Keep only cusp candidates that stay close to the most buccal candidate.

    This post-processing step lets watershed/local-extrema detection produce a
    broad cusp candidate set first, then removes lingual cusps using the tooth's
    buccolingual axis from the arch-curve frame.
    """

    def __init__(
        self,
        *,
        max_buccal_cusps: int = 3,
        buccal_distance_ratio: float = 0.1,
        buccal_distance_mm: float = 1.0,
        min_cusp_separation_ratio: float = 0.12,
        min_cusp_separation_mm: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_buccal_cusps = int(max_buccal_cusps)
        self.buccal_distance_ratio = float(buccal_distance_ratio)
        self.buccal_distance_mm = float(buccal_distance_mm)
        self.min_cusp_separation_ratio = float(min_cusp_separation_ratio)
        self.min_cusp_separation_mm = float(min_cusp_separation_mm)

    def _select_buccal_candidate_local_indices(
        self,
        context: ToothContext,
        candidate_local_indices: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        candidate_local_indices = np.asarray(candidate_local_indices, dtype=np.int64)
        if candidate_local_indices.size == 0:
            return np.empty(0, dtype=np.int64), 0.0

        candidate_buccolingual_values = np.asarray(
            context.buccolingual_values[candidate_local_indices],
            dtype=float,
        )
        order = np.argsort(candidate_buccolingual_values)[::-1]
        ordered_local_indices = candidate_local_indices[order]
        ordered_buccolingual_values = candidate_buccolingual_values[order]

        buccolingual_span = max(float(np.ptp(context.buccolingual_values)), 1e-6)
        distance_threshold = max(
            float(self.buccal_distance_mm),
            float(self.buccal_distance_ratio) * buccolingual_span,
        )

        anchor_value = float(ordered_buccolingual_values[0])
        selected_local_indices: list[int] = []
        for local_index, buccolingual_value in zip(
            ordered_local_indices,
            ordered_buccolingual_values,
            strict=False,
        ):
            if anchor_value - float(buccolingual_value) > distance_threshold:
                continue
            selected_local_indices.append(int(local_index))
            if len(selected_local_indices) >= self.max_buccal_cusps:
                break

        if not selected_local_indices:
            selected_local_indices = [int(ordered_local_indices[0])]

        return np.asarray(selected_local_indices, dtype=np.int64), float(distance_threshold)

    def _enforce_min_cusp_separation(
        self,
        context: ToothContext,
        candidate_local_indices: np.ndarray,
        keypoint_by_local_index: dict[int, object],
    ) -> tuple[np.ndarray, float, np.ndarray]:
        candidate_local_indices = np.asarray(candidate_local_indices, dtype=np.int64)
        if candidate_local_indices.size <= 1:
            return candidate_local_indices, 0.0, np.empty(0, dtype=np.int64)

        separation_threshold = max(
            float(self.min_cusp_separation_mm),
            float(self.min_cusp_separation_ratio) * float(context.horizontal_scale),
        )
        candidate_scores = np.asarray(
            [float(keypoint_by_local_index[int(local_index)].score) for local_index in candidate_local_indices],
            dtype=float,
        )
        candidate_buccolingual_values = np.asarray(
            context.buccolingual_values[candidate_local_indices],
            dtype=float,
        )
        ordered_indices = np.lexsort(
            (
                candidate_local_indices,
                -candidate_buccolingual_values,
                -candidate_scores,
            )
        )
        greedy_local_indices = candidate_local_indices[ordered_indices]

        kept_local_indices: list[int] = []
        removed_local_indices: list[int] = []
        for local_index in greedy_local_indices:
            local_index = int(local_index)
            local_point_2d = np.array(
                [
                    float(context.mesiodistal_values[local_index]),
                    float(context.buccolingual_values[local_index]),
                ],
                dtype=float,
            )
            too_close = False
            for kept_local_index in kept_local_indices:
                kept_point_2d = np.array(
                    [
                        float(context.mesiodistal_values[int(kept_local_index)]),
                        float(context.buccolingual_values[int(kept_local_index)]),
                    ],
                    dtype=float,
                )
                if np.linalg.norm(local_point_2d - kept_point_2d) < separation_threshold:
                    too_close = True
                    break
            if too_close:
                removed_local_indices.append(local_index)
                continue
            kept_local_indices.append(local_index)

        kept_local_indices_array = np.asarray(kept_local_indices, dtype=np.int64)
        if kept_local_indices_array.size == 0:
            kept_local_indices_array = np.asarray([int(greedy_local_indices[0])], dtype=np.int64)

        kept_buccolingual_values = np.asarray(context.buccolingual_values[kept_local_indices_array], dtype=float)
        kept_order = np.argsort(kept_buccolingual_values)[::-1]
        return (
            kept_local_indices_array[kept_order],
            float(separation_threshold),
            np.asarray(removed_local_indices, dtype=np.int64),
        )

    def recognize(self, context: ToothContext) -> ToothKeypointResult:
        raw_result = super().recognize(context)
        if not raw_result.keypoints:
            return raw_result

        local_lookup = {
            int(vertex_index): local_index
            for local_index, vertex_index in enumerate(np.asarray(context.vertex_indices, dtype=np.int64))
        }
        keypoint_by_local_index = {
            int(local_lookup[keypoint.vertex_index]): keypoint
            for keypoint in raw_result.keypoints
            if int(keypoint.vertex_index) in local_lookup
        }
        candidate_local_indices = np.asarray(sorted(keypoint_by_local_index), dtype=np.int64)
        if candidate_local_indices.size == 0:
            return raw_result

        selected_local_indices, distance_threshold = self._select_buccal_candidate_local_indices(
            context,
            candidate_local_indices,
        )
        pre_separation_local_indices = np.asarray(selected_local_indices, dtype=np.int64)
        selected_local_indices, separation_threshold, removed_close_local_indices = self._enforce_min_cusp_separation(
            context,
            selected_local_indices,
            keypoint_by_local_index,
        )
        anchor_local_index = int(
            candidate_local_indices[
                int(np.argmax(np.asarray(context.buccolingual_values[candidate_local_indices], dtype=float)))
            ]
        )
        anchor_buccolingual_value = float(context.buccolingual_values[anchor_local_index])

        filtered_keypoints = []
        for rank, local_index in enumerate(selected_local_indices):
            original_keypoint = keypoint_by_local_index[int(local_index)]
            buccolingual_value = float(context.buccolingual_values[int(local_index)])
            filtered_keypoints.append(
                replace(
                    original_keypoint,
                    kind=self.keypoint_kind(rank),
                    metadata={
                        **original_keypoint.metadata,
                        "selection_constraint": "buccal_side",
                        "buccolingual_value": buccolingual_value,
                        "buccal_anchor_value": anchor_buccolingual_value,
                        "buccolingual_distance_to_anchor": float(anchor_buccolingual_value - buccolingual_value),
                    },
                )
            )

        filtered_debug = dict(raw_result.debug)
        filtered_debug.update(
            {
                "raw_candidate_count": int(len(raw_result.keypoints)),
                "buccal_candidate_count": int(len(filtered_keypoints)),
                "max_buccal_cusps": int(self.max_buccal_cusps),
                "buccal_distance_threshold": float(distance_threshold),
                "buccal_anchor_local_index": int(anchor_local_index),
                "buccal_anchor_buccolingual_value": float(anchor_buccolingual_value),
                "pre_separation_buccal_local_indices": np.asarray(pre_separation_local_indices, dtype=np.int64),
                "selected_buccal_local_indices": np.asarray(selected_local_indices, dtype=np.int64),
                "min_cusp_separation_threshold": float(separation_threshold),
                "removed_close_local_indices": np.asarray(removed_close_local_indices, dtype=np.int64),
            }
        )

        return ToothKeypointResult(
            label=raw_result.label,
            tooth_name=raw_result.tooth_name,
            tooth_family=raw_result.tooth_family,
            recognizer_name=raw_result.recognizer_name,
            keypoints=filtered_keypoints,
            debug=filtered_debug,
        )
