import numpy as np
from scipy.stats import rv_continuous

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def uniform1d(n_x, n_y):
    """ 1d uniform probability density function (pdf) """

    return np.random.rand(n_x, n_y) - 0.5


def normal1d(n_x, n_y):
    """ 1d normal probability density function (pdf) """

    global FWHM_TO_SIGMA

    return FWHM_TO_SIGMA * np.random.randn(n_x, n_y)


def triangular1d(n_x, n_y):
    """ 1d triangular probability density function (pdf) """

    return np.random.triangular(-1, 0, 1, size=(n_x, n_y))


def uniform2d(n_x, n_y):
    """ 2d uniform probability density function (pdf) """

    # get random polar coordinates with uniform density
    r = np.sqrt(np.random.rand(n_x, n_y)) / 2
    phi = 2 * np.pi * np.random.rand(n_x, n_y)
    # convert polar to cartesian coordinates
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y


def normal2d(n_x, n_y):
    """ 2d normal probability density function (pdf) """

    global FWHM_TO_SIGMA

    return np.random.multivariate_normal([0, 0], [[FWHM_TO_SIGMA ** 2, 0], [0, FWHM_TO_SIGMA ** 2]], size=(n_x, n_y)).T


def triangular2d(n_x, n_y):
    """ 2d triangular probability density function (pdf) """

    # get the radial pdf required for the 2d triangular pdf
    triangular_radius = TriangularRadius(a=0, b=1, name="triangular")
    # get random polar coordinates with triangular density distribution
    r = triangular_radius.rvs(size=(n_x, n_y))
    phi = 2 * np.pi * np.random.rand(n_x, n_y)
    # convert the polar to cartesian coordinates
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return x, y


class TriangularRadius(rv_continuous):
    """ Class that is derived from scipy.stats.rv_continuous and allows to implement arbitrary distributions """

    # define the pdf for the radius of the triangular distribution
    def _pdf(self, x):
        return 6 * (x - x ** 2)
