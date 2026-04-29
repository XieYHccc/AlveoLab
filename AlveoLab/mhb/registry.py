from __future__ import annotations

from AlveoLab.mhb.labels import CANINE, INCISOR, MOLAR, PRIMARY_MOLAR, ToothLabelDefinition
from AlveoLab.mhb.recognizers import (
    BaseToothKeypointRecognizer,
    CanineMhbKeypointRecognizer,
    IncisorMhbKeypointRecognizer,
    MolarMhbKeypointRecognizer,
    PrimaryMolarMhbKeypointRecognizer,
)


class ToothRecognizerRegistry:
    def __init__(
        self,
        family_recognizers: dict[str, BaseToothKeypointRecognizer] | None = None,
        label_recognizers: dict[int, BaseToothKeypointRecognizer] | None = None,
    ):
        self.family_recognizers = family_recognizers or {
            INCISOR: IncisorMhbKeypointRecognizer(),
            CANINE: CanineMhbKeypointRecognizer(),
            PRIMARY_MOLAR: PrimaryMolarMhbKeypointRecognizer(),
            MOLAR: MolarMhbKeypointRecognizer(),
        }
        self.label_recognizers = label_recognizers or {}

    def register_family(self, family: str, recognizer: BaseToothKeypointRecognizer) -> None:
        self.family_recognizers[family] = recognizer

    def register_label(self, label: int, recognizer: BaseToothKeypointRecognizer) -> None:
        self.label_recognizers[int(label)] = recognizer

    def resolve(self, definition: ToothLabelDefinition) -> BaseToothKeypointRecognizer:
        if definition.label in self.label_recognizers:
            return self.label_recognizers[definition.label]
        if definition.family not in self.family_recognizers:
            raise KeyError(f"No recognizer registered for tooth family '{definition.family}'.")
        return self.family_recognizers[definition.family]

