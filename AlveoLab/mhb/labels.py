from dataclasses import dataclass


INCISOR = "incisor"
CANINE = "canine"
PRIMARY_MOLAR = "primary_molar"
MOLAR = "molar"


@dataclass(frozen=True)
class ToothLabelDefinition:
    label: int
    name: str
    side: str
    family: str
    default_keypoint_count: int


TOOTH_LABEL_DEFINITIONS: dict[int, ToothLabelDefinition] = {
    1: ToothLabelDefinition(1, "left_central_incisor", "left", INCISOR, 1),
    2: ToothLabelDefinition(2, "left_lateral_incisor", "left", INCISOR, 1),
    3: ToothLabelDefinition(3, "left_canine", "left", CANINE, 1),
    4: ToothLabelDefinition(4, "left_primary_first_molar", "left", PRIMARY_MOLAR, 4),
    5: ToothLabelDefinition(5, "left_primary_second_molar", "left", PRIMARY_MOLAR, 4),
    6: ToothLabelDefinition(6, "left_first_molar", "left", MOLAR, 4),
    7: ToothLabelDefinition(7, "right_central_incisor", "right", INCISOR, 1),
    8: ToothLabelDefinition(8, "right_lateral_incisor", "right", INCISOR, 1),
    9: ToothLabelDefinition(9, "right_canine", "right", CANINE, 1),
    10: ToothLabelDefinition(10, "right_primary_first_molar", "right", PRIMARY_MOLAR, 4),
    11: ToothLabelDefinition(11, "right_primary_second_molar", "right", PRIMARY_MOLAR, 4),
    12: ToothLabelDefinition(12, "right_first_molar", "right", MOLAR, 4),
}

