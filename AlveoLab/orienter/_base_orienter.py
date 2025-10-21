from abc import ABC, abstractmethod
import numpy as np
from trimesh import Trimesh
import AlveoLab.math.geometry as geom


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

    def __init__(self, mesh, arch_type=None):
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

    def to_horizontal(self, points) -> np.ndarray:
        """Extract the horizontal components of some **points**.

        More precisely, find the projections in the directions :attr:`right`
        and :attr:`forwards`.

        Args:
            points: A point, array of points, array of arrays of points etc.
        Returns:
            Horizontal components. An array with :py:`.shape[-1] == 2`.

        """
        return geom.get_components_zipped(points, self.right, self.forward)

    def from_horizontal(self, points_2d, up=None, occlusal=None):
        """Reconstruct points from their horizontal projections as returned
        by :meth:`to_horizontal`.

        Args:
            points_2d:
                The projections in the directions :attr:`right` and
                :attr:`forwards`. Should be an array with :py:`shape[-1] == 2`.
            up:
                The projection(s) in the :attr:`up` direction, defaults to
                ``0.0``.
            occlusal:
                The projection(s) in the :attr:`occlusal` direction,
                defaults to ``0.0``.
        Returns:
            Remapped points. An array with :py:`shape[-1] == 3`.

        More fine-grained control over what happens to the vertical axis can
        be achieved by feeding the output of this method to the
        :meth:`~motmot.geometry.UnitVector.with_` method of :attr:`up` or
        :attr:`occlusal`.

        """
        out = points_2d @ np.array([self.right, self.forward])
        if up is not None:
            out += self.up * np.array(up)[..., np.newaxis]
        if occlusal is not None:
            out += self.occlusal * np.array(occlusal)[..., np.newaxis]
        return out
