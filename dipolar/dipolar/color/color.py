import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import os


def xyz_from_xy(x, y):
    return np.array((x, y, 1-x-y))


class ColorSystem:
    """A class representing a color system.

    A color system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".

    The transformation from XYZ to rgb coordinates is described at
    http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    """
    # get the path of the cmf data
    script_dir = os.path.dirname(__file__)
    relative_path = 'cie_cmf.py'
    absolute_path = os.path.join(script_dir, relative_path)
    # the CIE color matching function for 380 - 780 nm in 5 nm intervals
    cmf = np.loadtxt(absolute_path)
    # extract the wavelength and convert (nm) to (µm)
    lambda_0 = cmf[:, 0] / 1e3
    # interpolate the color matching functions
    red_cmf = interp1d(lambda_0, cmf[:, 1], kind='cubic', bounds_error=False, fill_value=0)
    green_cmf = interp1d(lambda_0, cmf[:, 2], kind='cubic', bounds_error=False, fill_value=0)
    blue_cmf = interp1d(lambda_0, cmf[:, 3], kind='cubic', bounds_error=False, fill_value=0)

    def __init__(self, red, green, blue, white):
        # get XYZ matrices
        XYZ_red = [red[0] / red[1], 1, (1 - red[0] - red[1]) / red[1]]
        XYZ_green = [green[0] / green[1], 1, (1 - green[0] - green[1]) / green[1]]
        XYZ_blue = [blue[0] / blue[1], 1, (1 - blue[0] - blue[1]) / blue[1]]
        XYZ_white = [white[0] / white[1], 1, (1 - white[0] - white[1]) / white[1]]
        # chromaticities
        XYZ = np.vstack((XYZ_red, XYZ_green, XYZ_blue)).T
        XYZ_inv = np.linalg.inv(XYZ)
        # white scaling array
        S = np.dot(XYZ_inv, XYZ_white)
        # XYZ -> rgb transformation matrix
        self.tm = XYZ_inv / S[:, np.newaxis]

    def xyz_to_rgb(self, xyz):
        rgb = np.tensordot(xyz, self.tm.T, axes=(2, 0))

        return rgb

    def spec_to_xyz(self, lambda_0, intensity):
        # get the color matching functions in that wavelength range
        red_cmf = self.red_cmf(lambda_0)
        green_cmf = self.green_cmf(lambda_0)
        blue_cmf = self.blue_cmf(lambda_0)

        x_bar = np.tile(red_cmf, (intensity.shape[0], intensity.shape[1], 1))
        y_bar = np.tile(green_cmf, (intensity.shape[0], intensity.shape[1], 1))
        z_bar = np.tile(blue_cmf, (intensity.shape[0], intensity.shape[1], 1))

        norm = simps(y_bar[0, 0, :], lambda_0)

        X = simps(x_bar * intensity, lambda_0, axis=2) / norm
        Y = simps(y_bar * intensity, lambda_0, axis=2) / norm
        Z = simps(z_bar * intensity, lambda_0, axis=2) / norm

        XYZ = np.dstack((X, Y, Z))

        return XYZ

    def spec_to_rgb(self, lambda_0, intensity, gamma=True):
        xyz = self.spec_to_xyz(lambda_0, intensity)
        rgb = self.xyz_to_rgb(xyz)

        # gamma correct the intensities (srgb gamma correction)
        if gamma:
            # normalize the values
            rgb /= np.nanmax(rgb)
            srgb = rgb.copy()
            # apply the gamma correction
            thresh = 0.0031308
            srgb[rgb <= thresh] = 12.92 * rgb[rgb <= thresh]
            srgb[rgb > thresh] = 1.055 * rgb[rgb > thresh] ** (1 / 2.4) - 0.055
            # rename the srgb colorspace
            rgb = srgb

        if np.any(rgb < 0):
            rgb[rgb < 0] = 0
        if not np.all(rgb == 0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)

        return rgb