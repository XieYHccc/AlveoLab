import numpy as np
from numpy.polynomial import Polynomial

from AlveoLab.math.geometry import normalize_vector, real_and_bounded, stagger, inner_product
from AlveoLab.utils import LazyAttribute, zip_axes, unzip_axes, copy_name_wrapper
from AlveoLab.kdtree import KDTree


class QuadraticFit(object):
    def __init__(self, x, y, weights=None):
        self.points = np.array([x, y]).T
        self.x, self.y = self.points.T
        self.weights = weights
        self.quadratic = Polynomial.fit(self.x, self.y, deg=2, w=weights).convert()

        self.derivative = self.quadratic.deriv()

    def sort(self):
        args = np.argsort(self.nearest_points_on_quadratic[:, 0])
        for attr in [
                "points",
                "x",
                "y",
                "buccals",
                "distals",
                "nearest_points_on_quadratic",
        ]:
            setattr(self, attr, getattr(self, attr)[args])
        return args

    def nearest_point_on_quadratic(self, x, y):
        assert np.isscalar(y)
        if not (np.isfinite(x) and np.isfinite(y)):
            return np.array([np.nan, np.nan])
        dist_sqr_poly = (self.quadratic - y)**2 + (x - np.polynomial.Polynomial.identity())**2
        dist_sqr_deriv = dist_sqr_poly.deriv()
        stationary_xs = real_and_bounded(dist_sqr_deriv.roots(), -np.inf, np.inf)
        stationary_ys = self.quadratic(stationary_xs)
        arg = np.argmin((stationary_xs - x)**2 + (stationary_ys - y)**2)
        return stationary_xs[arg], stationary_ys[arg]

    def tangent_at(self, x, y=None):
        if y is not None:
            x = self.nearest_point_on_quadratic(x, y)[0]

        out = np.empty(np.shape(x) + (2,))
        out[..., 0] = 1
        out[..., 1] = self.derivative(x)
        normalize_vector(out)
        return out

    def distal_at(self, x, y=None):
        tangent = self.tangent_at(x, y)
        tangent *= np.sign(tangent[..., 1] * self.quadratic.coef[2])[..., np.newaxis]
        return tangent

    def buccal_at(self, x, y=None):
        if y is not None:
            x = self.nearest_point_on_quadratic(x, y)[0]

        out = np.empty(np.shape(x) + (2,), float)
        out[..., 0] = self.derivative(x)
        out[..., 1] = -1
        out *= np.sign(self.quadratic.coef[2])
        normalize_vector(out)
        return out

    @LazyAttribute
    def tip(self):
        x = -self.quadratic.coef[1] / (2 * self.quadratic.coef[2])
        return np.array([x, self.quadratic(x)])

    @LazyAttribute
    def height(self):
        xs = self.points[:, 0]
        return self.tip[1] - min(self(xs.min()), self(xs.max()))

    def __call__(self, x):
        return self.quadratic(x)

    def get_distance_between_points(self, x0, x1, y0, y1, n=50):
        t0 = self.nearest_point_on_quadratic(x0, y0)[0]
        t1 = self.nearest_point_on_quadratic(x1, y1)[0]
        t = np.linspace(t0, t1, n)

        r0 = inner_product([x0 - t0, y0 - self(t0)], normalize_vector(self.buccal_at(t0)))
        r1 = inner_product([x1 - t1, y1 - self(t1)], normalize_vector(self.buccal_at(t1)))

        dt = t1 - t0
        radii = (t1 - t) / dt * r0 + (t - t0) / dt * r1
        buccals = self.buccal_at(t)
        tangents = self.tangent_at(sum(stagger(t)) / 2)
        x = t + radii * buccals[:, 0]
        y = self(t) + radii * buccals[:, 1]
        #        plt.plot(x, y)
        #        plt.show()
        lengths = np.diff(x) * tangents[:, 0] + np.diff(y) * tangents[:, 1]

        return np.sum(lengths)

    # def plot_peak_points(self):
    #     import matplotlib.pylab as plt
    #
    #     plt.scatter(self.x, self.y)
    #     plt.scatter(*self.tip)
    #     plt.plot(*self.nearest_points_on_quadratic.T, c="b")
    #     plt.plot(
    #         [self.x, self.nearest_points_on_quadratic[:, 0]],
    #         [self.y, self.nearest_points_on_quadratic[:, 1]],
    #         c="r",
    #     )
    #
    # def plot_buccals(self):
    #     import matplotlib.pylab as plt
    #
    #     plt.quiver(self.x, self.y, *self.buccals.T)
    #
    # def plot_distals(self):
    #     import matplotlib.pylab as plt
    #
    #     plt.quiver(self.x, self.y, *self.distals.T, color="g")
    #
    # def plot(self):
    #     import matplotlib.pylab as plt
    #
    #     plt.axes().set_aspect("equal", "datalim")
    #     self.plot_peak_points()
    #     self.plot_buccals()
    #     self.plot_distals()
    #     plt.show()
    #
    # def plot_separate(self):
    #     import matplotlib.pylab as plt
    #
    #     plt.axes().set_aspect("equal", "datalim")
    #     self.plot_peak_points()
    #     #plt.savefig("least_squares_quadratic_0.pdf")
    #     plt.show()
    #
    #     plt.axes().set_aspect("equal", "datalim")
    #     self.plot_peak_points()
    #     self.plot_distals()
    #     #plt.savefig("least_squares_quadratic_1.pdf")
    #     plt.show()
    #
    #     plt.axes().set_aspect("equal", "datalim")
    #     self.plot_peak_points()
    #     self.plot_buccals()
    #     #plt.savefig("least_squares_quadratic_2.pdf")
    #     plt.show()

    def get_distance_map(self, points=None):
        if points is None:
            points = self.points

        to_neighbour = np.array([
            self.get_distance_between_points(x0, x1, y0, y1)
            for ((x0, y0), (x1, y1)) in zip(*stagger(points))
        ])

        cs_1d = np.empty(len(points), points.dtype)
        cs_1d[0] = 0
        cs_1d[1:] = np.cumsum(to_neighbour)

        dist_map = cs_1d - cs_1d[:, np.newaxis]
        return dist_map


class FuzzyQuadraticFit(QuadraticFit):
    def __init__(self, x, y, weights=None, resolution=200):
        super().__init__(x, y, weights)
        self.resolution = resolution
        self.tree_points = np.empty((resolution, 2))
        xs = self.tree_points[:, 0] = np.linspace(x.min(), x.max(), resolution)
        self.tree_points[:, 1] = self(self.tree_points[:, 0])
        self.tree = KDTree(self.tree_points)
        self.nearest_points = self.tree_points[self.tree.query(self.points)[1]]

        distances = inner_product(self.buccal_at(self.nearest_points[:, 0]),
                                       self.points - self.nearest_points)
        sort_args = self.nearest_points[:, 0].argsort()
        wiggly_points = (
            self.tree_points + self.buccal_at(xs) *
            np.interp(xs, self.nearest_points[sort_args, 0], distances[sort_args])[:, np.newaxis])

        self.lengths = np.empty_like(wiggly_points[:, 0])
        self.lengths[0] = 0
        inner_product(np.diff(wiggly_points, axis=0), self.tangent_at(
            (xs[:-1] + xs[1:]) / 2)).cumsum(out=self.lengths[1:])

    def get_distance_map(self, points=None):
        if points is None:
            points = self.points
        # TODO: sort this so that out of range points are handles properly.
        lengths = self.lengths[self.tree.query(points)[1]]
        return lengths - lengths[:, np.newaxis]

    def nearest_point_on_quadratic(self, x, y):
        args = self.tree.query(zip_axes(x, y))[1]
        return unzip_axes(self.tree_points[args])

    def get_distance_between_points(self, x0, x1, y0, y1, n=50):
        i, j = self.tree.query([[x0, y0], [x1, y1]])[1]
        return -self.lengths[i] + self.lengths[j]


class Quadratic3D(object):
    def __init__(self, points, orienter, weights=None, fuzzy_resolution=None):
        assert len(points)
        self.orienter = orienter
        self.to_2d = self.orienter.to_horizontal
        self.points_2d = self.to_2d(points)

        x, y = self.points_2d.T

        if fuzzy_resolution is None:
            self.quadratic_2d = QuadraticFit(x, y, weights=weights)
        else:
            self.quadratic_2d = FuzzyQuadraticFit(*self.points_2d, weights=weights,
                                                  resolution=fuzzy_resolution)

        self.height = self.orienter.up * inner_product(self.orienter.up, np.mean(points, 0))
        self.points = points

        #self.distals = self.to_3d(*self.quadratic_2d.distals.T)
        #self.buccals = self.to_3d(*self.quadratic_2d.buccals.T)

    def sort(self):
        reorder_args = self.quadratic_2d.sort()
        for attr in ["points", "buccals", "distals"]:
            setattr(self, attr, getattr(self, attr)[reorder_args])
        return reorder_args

    def to_3d(self, points_2d, add_z=False):
        out = self.orienter.from_horizontal(points_2d)
        if add_z:
            out += self.height
        return out

    @copy_name_wrapper
    def _from_2d_method(unbound_method_2d):
        def method_3d(self, point=None, root=None):
            if (root is None) == (point is None):
                raise ValueError("Exactly one of `point` and `root` arguments must be specified")
            if root is None:
                root = self.get_root_at(point)
            return self.to_3d(unbound_method_2d(self.quadratic_2d, root))

        method_3d.__name__ = unbound_method_2d.__name__
        method_3d.__qualname__ = unbound_method_2d.__qualname__

        return method_3d

    tangent_at = _from_2d_method(QuadraticFit.tangent_at)
    distal_at = _from_2d_method(QuadraticFit.distal_at)
    buccal_at = _from_2d_method(QuadraticFit.buccal_at)

    def get_root_at(self, point):
        return self.quadratic_2d.nearest_point_on_quadratic(*self.to_2d(point))[0]

    # def plot(self, z=None, **plotargs):
    #     x = np.linspace(self.points_2d[0].min(), self.points_2d[0].max())
    #
    #     points = self.to_3d(x, self.quadratic_2d(x), add_z=True)
    #
    #     if z is not None:
    #         points = self.odom.occlusal.with_projection(points, z)
    #
    #     vpl.plot(points, **plotargs)
    #
    # def plot_distals(self, **plotargs):
    #     vpl.quiver(self.points, self.distals, label="Distal", **plotargs)
    #
    # def plot_buccals(self, **plotargs):
    #     vpl.quiver(self.points, self.buccals, label="Buccal", **plotargs)

    def __call__(self, t):
        return self.to_3d(t, self.quadratic_2d(t))

    def nearest_point_on_quadratic(self, point):
        t = self.get_root_at(point)
        nearest = self(t)
        return nearest

    # def make_odom(self, centre_of_mass):
    #     """Build a :class:`ToothOdometry` object based on its position given by
    #     **centre_of_mass** and orientations derived from this quadratic.
    #     """
    #     from ALR import mesh_orientation
    #
    #     root = self.get_root_at(centre_of_mass)
    #     return mesh_orientation.ToothOdometry(
    #         self.distal_at(root=root),
    #         self.buccal_at(root=root),
    #         self.odom.occlusal,
    #         centre_of_mass,
    #     )