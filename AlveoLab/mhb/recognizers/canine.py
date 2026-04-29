from AlveoLab.mhb.recognizers.cusp_based import CuspBasedMhbKeypointRecognizer


class CanineMhbKeypointRecognizer(CuspBasedMhbKeypointRecognizer):
    name = "canine"

    def __init__(self, **kwargs):
        super().__init__(max_candidates=1, occlusal_quantile=0.82, **kwargs)

    def keypoint_kind(self, rank: int) -> str:
        return "cusp_tip"
