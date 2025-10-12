from abc import ABC, abstractmethod
import numpy as np
from trimesh import Trimesh


class BaseOrienter(ABC):
    """
    Abstract base class for orientation of trimesh dental models.

    +-------------------+------------------------------------------------------+
    | Attribute         | Description                                          |
    +===================+======================================================+
    | :attr:`right`     | Both from the doctor's perspective.                 |
    +-------------------+                                                      |
    | :attr:`forwards`  |                                                      |
    +-------------------+------------------------------------------------------+
    | :attr:`up`        | To the roof regardless of                          |
    |                   | whether the model is maxillary of mandibular.        |
    +-------------------+------------------------------------------------------+
    | :attr:`occlusal`  | Alias for the direction of the teeth. Up if it is    |
    |                   | a lower jaw or down if it is an upper jaw.           |
    +-------------------+------------------------------------------------------+
    """

    mesh: Trimesh
    arch_type: str
    """Either of:

    * :py:`'U'` for a maxillary (upper) jaw.
    * :py:`'L'` for a mandibular (lower) jaw.

    """

    def __init__(self, mesh, arch_type = None):
        self.mesh = mesh
        self.arch_type = arch_type

    @property
    @abstractmethod
    def center(self):
        """Center of Mass of the mesh."""
        pass

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
    def occlusal(self):
        """:attr:`up` for a mandibular model, down for a maxillary model."""
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
