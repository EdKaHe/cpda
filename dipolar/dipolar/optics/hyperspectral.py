import numpy as np

from ..color.color import ColorSystem, xyz_from_xy
from scipy.constants import mu_0, epsilon_0
from scipy.interpolate import interp1d
from scipy.integrate import simps

# define constants
IDENTITY_MATRIX = np.array([[1, 0],
                            [0, 1]])
Z_0 = np.sqrt(mu_0 / epsilon_0)


class Hyperspectral:
    def __init__(self, x, y, lambda_0, E_xx, E_xy, E_yx, E_yy):
        """ Hyperspectral class

            public attributes:
                x (array): spatial 2d array of positions in x-direction
                y (array): spatial 2d array of positions in y-direction
                lambda_0 (array): 1d array that specifies the wavelength
                E_xx, E_xy, E_yx, E_yy (array): 3d array of the hyperspectral electric fields with first dimension being the
                    electric fields along the x-axis, the second dimension being the electric fields along the y-axis,
                    and the third axis being the electric fields along the wavelength axis
                block (None or tuple): if None all the intensity data is shown otherwise the tuple elements indicate the
                    inner and outer radii that are removed from the intensity data

            public methods:
                intensity: returns 3d array of the hyperspectral intensities with same format as the electric fields
                spectral: returns 1d array with spectral intensities
                spatial: returns 2d array with spatial intensities

            protected methods:
                _jones_formalism (static): returns two 2d arrays of x- and y-polarized electric fields in dependence
                    of the analyzer and polarizer state
        """

        # initialize protected attributes
        self._x = x
        self._y = y
        self._lambda_0 = lambda_0.reshape((-1,))
        self._E_xx = E_xx
        self._E_xy = E_xy
        self._E_yx = E_yx
        self._E_yy = E_yy
        self.block = None

    # getter for the x-component
    @property
    def x(self):
        return self._x

    # getter for the y-component
    @property
    def y(self):
        return self._y

    # getter for the wavelength
    @property
    def lambda_0(self):
        return self._lambda_0

    # getter for the xx-components
    @property
    def E_xx(self):
        return self._E_xx

    # getter for the xy-components
    @property
    def E_xy(self):
        return self._E_xy

    # getter for the yx-components
    @property
    def E_yx(self):
        return self._E_yx

    # getter for the yy-components
    @property
    def E_yy(self):
        return self._E_yy

    # getter for the intensity
    def intensity(self, jones_pol=None, jones_pre=None, jones_post=None):
        # get the spatial coordinates
        x = self.x
        y = self.y

        if jones_pre is None:
            jones_pre = IDENTITY_MATRIX
        if jones_post is None:
            jones_post = IDENTITY_MATRIX
        if jones_pol is None:
            # get the jones vectors for 0 and 90 degree polarized light
            jones_pol_0 = np.array([1, 0])
            jones_pol_90 = np.array([0, 1])

            # get the x- and y-components of the outgoing fields
            E_x_0, E_y_0 = Hyperspectral._jones_formalism(self.E_xx, self.E_xy, self.E_yx, self.E_yy,
                                                          jones_pol=jones_pol_0, jones_pre=jones_pre,
                                                          jones_post=jones_post)
            E_x_90, E_y_90 = Hyperspectral._jones_formalism(self.E_xx, self.E_xy, self.E_yx, self.E_yy,
                                                            jones_pol=jones_pol_90, jones_pre=jones_pre,
                                                            jones_post=jones_post)

            # calculate the intensity for unpolarized input from the electric fields
            intensity = (np.abs(E_x_0) ** 2 + np.abs(E_x_90) ** 2 + np.abs(E_y_0) ** 2 + np.abs(E_y_90) ** 2) / 2
        else:
            # get the x- and y-components of the outgoing fields
            E_x, E_y = Hyperspectral._jones_formalism(self.E_xx, self.E_xy, self.E_yx, self.E_yy,
                                                      jones_pol=jones_pol, jones_pre=jones_pre, jones_post=jones_post)

            # calculate the intensity from the electric fields
            intensity = np.abs(E_x) ** 2 + np.abs(E_y) ** 2

        # block the zeroth order
        if self.block:
            mask = (x ** 2 + y ** 2 <= self.block[0] ** 2) | \
                   ((x ** 2 + y ** 2 >= self.block[1] ** 2) & (x ** 2 + y ** 2 <= 1))
            intensity[mask, :] = np.nan

        return intensity / (2 * Z_0)

    # get the spectral intensities
    def spectral(self, jones_pol=None, jones_pre=None, jones_post=None):
        # get the hyperspectral intensity
        intensity = self.intensity(jones_pol=jones_pol, jones_pre=jones_pre, jones_post=jones_post)
        # set nans to zero
        intensity[np.isnan(intensity)] = 0
        # integrate the intensity along the spatial axes
        spectral = simps(simps(intensity, self._x[0, :], axis=0), self._y[:, 0], axis=0)

        return spectral.reshape((-1,))

    # get the spatial intensities or rgb colors
    def spatial(self, mode='int', lambda_i=None, jones_pol=None, jones_pre=None, jones_post=None):
        # get the wavelength
        lambda_0 = self.lambda_0
        # get the hyperspectral intensity
        intensity = self.intensity(jones_pol=jones_pol, jones_pre=jones_pre, jones_post=jones_post)

        # get the indices where all entries along the 3rd axis are nan
        nan_idx = np.all(np.isnan(intensity), axis=2)
        # replace all nans by zeros
        intensity[np.isnan(intensity)] = 0

        # check which mode to choose
        if mode == 'int':
            # sum the intensity along the 3rd axis
            if lambda_i is None:
                spatial = np.sum(intensity, axis=2)
            else:
                if np.size(lambda_0) > 1:
                    spatial_interp = interp1d(lambda_0, intensity, axis=2)
                    intensity = spatial_interp(lambda_i)
                    if intensity.ndim > 2:
                        spatial = np.sum(intensity, axis=2)
                    else:
                        spatial = intensity.copy()
                else:
                    spatial = np.squeeze(intensity)

            # reset all entries to nan
            spatial[nan_idx] = np.nan
        elif mode == 'rgb':
            # replace all nans by zeros
            intensity[np.isnan(intensity)] = 0

            # define the linear rgb color system with equal energy illumination
            cs = ColorSystem(red=xyz_from_xy(0.64, 0.33),
                             green=xyz_from_xy(0.3, 0.6),
                             blue=xyz_from_xy(0.15, 0.06),
                             white=xyz_from_xy(1 / 3, 1 / 3))
            # cs = ColorSystem(red=xyz_from_xy(0.735, 0.265),
            #                  green=xyz_from_xy(0.274, 0.717),
            #                  blue=xyz_from_xy(0.167, 0.009),
            #                  white=xyz_from_xy(1 / 3, 1 / 3))

            spatial = cs.spec_to_rgb(lambda_0, intensity)

            # add alpha channel
            spatial = np.dstack((spatial, np.ones(spatial.shape[:2])))

            # make nan values transparent
            spatial[nan_idx, 3] = 0
        else:
            raise ValueError('mode must be "int" or "rgb"')

        return spatial

    # apply the jones formalism to consider influence of optical elements before and after the sample
    @staticmethod
    def _jones_formalism(E_xx, E_xy, E_yx, E_yy, jones_pol, jones_pre, jones_post):
        # get the outgoing x-polarized fields
        E_x = (jones_post[0, 0] * E_xx + jones_post[0, 1] * E_xy) \
              * (jones_pre[0, 0] * jones_pol[0] + jones_pre[0, 1] * jones_pol[1]) \
              + (jones_post[0, 0] * E_yx + jones_post[0, 1] * E_yy) \
              * (jones_pre[1, 0] * jones_pol[0] + jones_pre[1, 1] * jones_pol[1])
        # get the outgoing y-polarized fields
        E_y = (jones_post[1, 0] * E_xx + jones_post[1, 1] * E_xy) \
              * (jones_pre[0, 0] * jones_pol[0] + jones_pre[0, 1] * jones_pol[1]) \
              + (jones_post[1, 0] * E_yx + jones_post[1, 1] * E_yy) \
              * (jones_pre[1, 0] * jones_pol[0] + jones_pre[1, 1] * jones_pol[1])

        return E_x, E_y
