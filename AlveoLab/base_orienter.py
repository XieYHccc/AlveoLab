from abc import ABC, abstractmethod
import numpy as np
from trimesh import Trimesh


class BaseOrienter(ABC):
    """
    Abstract base class for orientation of dental models.
    """

    def __init__(self, mesh: Trimesh, arch_type=None):
        self.mesh = mesh
        self.arch_type = arch_type

    @property
    @abstractmethod
    def up(self):
        """The direction of the teeth, point up if it is a lower jaw or down if it is an upper jaw."""
        pass

    @property
    @abstractmethod
    def forward(self):
        """Pointing out from mouse."""
        pass

    @property
    @abstractmethod
    def right(self):
        """The cross product of up and forward."""
        pass

    @property
    @abstractmethod
    def axes(self) -> np.ndarray:
        """3x3 matrix of core unit vectors. actually just the transpose of to_origin_transform_matrix"""
        pass

    @property
    @abstractmethod
    def to_origin_transform_matrix(self):
        """Transformation matrix to move the center of the bounding box to the origin."""
        pass
