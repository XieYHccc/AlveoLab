# This file is part of the ALR project: https://github.com/bwoodsend/ALR/
# It is distributed under GPL 3.0: https://www.gnu.org/licenses/gpl-3.0.html

import numpy as np

class UnBoolable(object):
    """Use instead of ``None`` as a placeholder for an uninitialised boolean variable."""
    def __bool__(self):
        raise ValueError("I am a non-binary object!!!")

    def __repr__(self):
        return "?"


UN_BOOLABLE = UnBoolable()


class Peak:
    def __init__(self, point, index):
        self.point = point
        self.index = index
        self.buccal = None
        self.distal = None
        self.occlusal = None
        self.spilled = UN_BOOLABLE

    def __repr__(self):
        return "Peak {} at {} spilled={})".format(self.index, self.point.round(2), self.spilled)

    def __str__(self):
        return str(self.index)
