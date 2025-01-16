import numpy as np
from trimesh import Trimesh

from AlveoLab.orienter._base_orienter import BaseOrienter
from AlveoLab.orienter._pca import Pca
from AlveoLab.geometry import normalize_vector


class PcaOrienter(BaseOrienter):
    """
    Find the orientation of a dental mesh model by computing the mesh's oriented bounding box
    """

    @property
    def right(self):
        return self._axisX

    @property
    def up(self):
        return self._axisY

    @property
    def forward(self):
        return self._axisZ

    @property
    def center(self):
        return self._center

    @property
    def axes(self) -> np.ndarray:
        """
        The core unit-vectors listed in :attr:`names` as columns of a
        3x3 matrix.

        This rotation matrix may be used to normalise and unnormalise a set of
        points.
            # Transform points(3, n) into a simplified coordinate system.
            normalised = odom.axes.T @ points
            # Get back to the original coordinate system using:
            points =  odom.axes @ normalised
        """
        return np.column_stack((self._axisX, self._axisY, self._axisZ))

    @property
    def to_origin_transform_matrix(self):
        """
        Transformation matrix which will move the center of the
        bounding box of the input mesh to the origin.

        normalized_mesh = to_origin_transform_matrix @ mesh
        """

        result = np.eye(4)
        result[:3, :3] = self.axes.T
        result[:3, 3] = -self.center
        return result

    def __init__(self, mesh: Trimesh, arch_type=None):
        super().__init__(mesh, arch_type)  # Initialize common attributes in the base class
        self._axisX = None
        self._axisY = None
        self._axisZ = None
        self._to_origin_transform_matrix = None
        self._center = None
        self._run()

    def _run(self):
        """
        Run all steps.

        I have written each function in the order they get used, so reading
        this process should just be a case of scrolling down through this
        class
        """
        self._apply_pca()
        self._check_axis_z_sign()
        self._check_axis_y_sign()
        self._check_axis_x_sign()

    def _apply_pca(self):
        """
        Run PCA to get the shortest, middlemost and longest axes.

        I'm choosing to define the axes using engineering convention:

        * eX  left -> right (patient's left and right)
        * eY  back -> front (going out of the patients mouth)
        * eZ  bottom -> top

        A dental model is longer than it is tall and is wider than it is long.
        So in ascending order of covariance (as numpy.linalg.eigh returns) they
        should be eZ, eY, eX.
        """
        convex_hull = self.mesh.convex_hull
        weights = (.05 - self.mesh.area_faces).clip(min=0)
        pca = Pca(self.mesh.triangles_center, weights)
        self._center = pca.center_of_mass
        self._axisY = normalize_vector(pca.eigenvectors[0])
        self._axisZ = normalize_vector(pca.eigenvectors[1])
        self._axisX = normalize_vector(pca.eigenvectors[2])

    def _check_axis_y_sign(self):
        """
        check/corrct the sign of the vertical axis

        The triangle density is much higher on the occlusal surface so a mean
        of mesh.face_normals could give a decent approximation of occlusal
        """

        # get an approximate up direction from the mesh's face normals.
        approximated_up = normalize_vector([i.sum() for i in self.mesh.face_normals.T])

        # compare it with current up direction
        agreement = np.dot(approximated_up, self._axisY)

        self._axisY *= np.sign(agreement)  # swap sign if necessary

    def _check_axis_z_sign(self):
        """
        Check/correct the sign of the forwards/backwards axis.

        Fit a weighted quadratic curve to the horizontal components of every
        mesh polygons' center. This curve will approximate the jaw line. If
        the sign is correct, curve should be ⋂ shaped (negative x² coefficient).
        If it is ⋃ shaped then the y-axis needs flipping.

        Note that the fit is quite poor and shouldn't be used for anything
        precise.
        """

        # extract the horizontal components, removing the center of mass
        x, y = (np.dot((self.mesh.triangles_center - self.center), e)
                for e in (self._axisX, self._axisZ))

        # prioritise the more occlusal points
        weights = np.dot(self.mesh.triangles_center, self._axisY)

        # prioritise non-occlusal facing triangles.
        # This is supposed to capture the labial and lingual vertical surfaces.
        # crosses = np.cross(self.mesh.face_normals, self._axisY)
        # weights *= np.dot(crosses ** 2, [1] * crosses.shape[1])

        weights -= np.min(weights)
        weights = weights ** 5

        # Fit a quadratic curve to the points with a weighted fitting.
        poly = np.polynomial.Polynomial.fit(x, y, 2, w=weights)

        # If x² coefficient is positive:
        if poly.convert().coef[2] > 0:
            # Flip eY
            self._axisZ = -self._axisZ

    def _check_axis_x_sign(self):
        """
        Finally eX is just determined so as not to mirror the mesh. This must
        be done after checking eZ and eY because it uses them.
        """

        # If rotation matrix mirrors then reverse eX
        self._axisX *= np.sign(np.linalg.det(self.axes))