import numpy as np
from numpy.polynomial import Polynomial

from AlveoLab.geometry import normalize_vector, real_and_bounded
from AlveoLab.utils import LazyAttribute


class QuadraticFit(object):
    def __init__(self, x, y, weights=None):
        self.points = np.array([x, y]).T
        self.x, self.y = self.points.T
        self.weights = weights
        self.quadratic = Polynomial.fit(self.x, self.y, deg=2, w=weights).convert()

        self.derivative = self.quadratic.deriv()
        self._nearest_point_on_quadratic_cache = {}

        #self.tip = self.get_tip()
        #self.height = self.get_height()

        #for (i, (x, y)) in enumerate(self.points):
        #    self.nearest_points_on_quadratic[i] = self.nearest_point_on_quadratic(x, y)
        #        #self.nearest_points_on_quadratic = np.empty_like(self.points)

        #self.buccals = self.buccal_at(self.nearest_points_on_quadratic[..., 0])
        #self.distals = self.distal_at(self.nearest_points_on_quadratic[..., 0])

        #self.sort()

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

    #@nuts_and_bolts.cached("_nearest_point_on_quadratic_cache")

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

        r0 = geom.inner_product([x0 - t0, y0 - self(t0)], geom.normalised(self.buccal_at(t0)))
        r1 = geom.inner_product([x1 - t1, y1 - self(t1)], geom.normalised(self.buccal_at(t1)))

        dt = t1 - t0
        radii = (t1 - t) / dt * r0 + (t - t0) / dt * r1
        buccals = self.buccal_at(t)
        tangents = self.tangent_at(sum(geom.stagger(t)) / 2)
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
            for ((x0, y0), (x1, y1)) in zip(*geom.stagger(points))
        ])

        cs_1d = np.empty(len(points), points.dtype)
        cs_1d[0] = 0
        cs_1d[1:] = np.cumsum(to_neighbour)

        dist_map = cs_1d - cs_1d[:, np.newaxis]
        return dist_map
