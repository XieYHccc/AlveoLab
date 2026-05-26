"""Tooth-id → RGB palette.

Index 0 is gum / gingiva. Indices 1..12 are the twelve tooth colors used by
`AlveoLab.visualization.LandmarkRecognizerVisualization.add_mesh_with_labels`,
lifted here so the client does not depend on that class. Mapping beyond 12 wraps
via `palette_for_labels`.
"""

from __future__ import annotations

import numpy as np


# Float RGB in [0, 1]. Row 0 is gum.
PALETTE: np.ndarray = (
    np.array(
        [
            [220, 220, 220],  # 0 - gum / gingiva (light grey)
            [153, 76, 0],
            [153, 153, 0],
            [76, 153, 0],
            [0, 153, 153],
            [0, 0, 153],
            [153, 0, 153],
            [255, 128, 0],
            [153, 153, 0],
            [76, 153, 0],
            [0, 153, 153],
            [0, 0, 153],
            [153, 0, 153],
        ],
        dtype=float,
    )
    / 255.0
)
# Dim the second half so that wrap-around hues stay distinguishable from 1..6.
PALETTE[7:] *= 0.55


def palette_for_labels(labels: np.ndarray) -> np.ndarray:
    """Map an int label array to an `(N, 3)` float RGB array in [0, 1].

    `0` always renders as gum (PALETTE[0]). Positive labels wrap around the
    12-tooth color band: `k -> PALETTE[((k - 1) % 12) + 1]`. Negative or
    unexpected labels fall back to the gum color.
    """
    labels = np.asarray(labels, dtype=np.int64)
    out = np.empty((labels.shape[0], 3), dtype=float)
    out[:] = PALETTE[0]
    positive = labels > 0
    if positive.any():
        idx = ((labels[positive] - 1) % 12) + 1
        out[positive] = PALETTE[idx]
    return out
