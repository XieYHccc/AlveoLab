from AlveoLab.mhb.recognizers.buccal_filtered_cusp import BuccalFilteredCuspRecognizer


class MolarMhbKeypointRecognizer(BuccalFilteredCuspRecognizer):
    name = "molar"

    def __init__(
        self,
        max_candidates: int = 5,
        *,
        max_buccal_cusps: int = 3,
        buccal_distance_ratio: float = 0.18,
        buccal_distance_mm: float = 1.0,
        min_cusp_separation_ratio: float = 0.13,
        min_cusp_separation_mm: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            max_candidates=max_candidates,
            occlusal_quantile=0.8,
            max_buccal_cusps=max_buccal_cusps,
            buccal_distance_ratio=buccal_distance_ratio,
            buccal_distance_mm=buccal_distance_mm,
            min_cusp_separation_ratio=min_cusp_separation_ratio,
            min_cusp_separation_mm=min_cusp_separation_mm,
            **kwargs,
        )
