import numpy as np

from AlveoLab.math.geometry import normalize_vector, inner_product


class ToothOrienter:
    def __init__(self, arch_quadratic, obb):
        self.arch_quadratic = arch_quadratic
        self.obb = obb
        self.center = obb.center

        self.axes = obb.to_origin_matrix[:3, :3]  # each column is an axis

        root = arch_quadratic.get_root_at(self.center)
        arch_distal = arch_quadratic.distal_at(root)
        for i in range(3):
            axis_vector = self.axes[i]
            inner_with_occlusal = inner_product(axis_vector, arch_quadratic.orienter.occlusal)
            inner_with_distal = inner_product(axis_vector, arch_distal)
            if abs(inner_with_occlusal) > 0.8:
                self.occlusal = normalize_vector(axis_vector) * (1 if np.sign(inner_with_occlusal) > 0 else -1)
            elif abs(inner_with_distal) > 0.8:
                self.distal = normalize_vector(axis_vector) * (1 if np.sign(inner_with_distal) > 0 else -1)
            else:
                self.buccal = normalize_vector(axis_vector)

        assert hasattr(self, 'occlusal')
        assert hasattr(self, 'distal')
        assert hasattr(self, 'buccal')
