#!/usr/bin/env python

"""

"""

from .ensemble import Ensemble
from .hyperspectral import Hyperspectral

import numpy as np
import numpy.matlib
from scipy.spatial import distance
from scipy.constants import epsilon_0
from scipy.interpolate import RectBivariateSpline
from warnings import warn

# define constants
EPSILON_0 = epsilon_0 * 1e-6  # [A s / (V µm)]


class Simulator:
    # private class attributes
    __px_x = 2 ** 9
    __px_y = 2 ** 9
    __eps = 2.25
    __mu = 1.0
    __na = 1.0
    __phi = 0.0
    __theta = 0.0
    __dz = 0.0
    __delta_image = 1.0
    __delta_fourier = 1.0

    def __init__(self, ensemble, lambda_0,
                 px_x=__px_x, px_y=__px_y,
                 eps=__eps, mu=__mu, na=__na,
                 phi=__phi, theta=__theta, dz=__dz,
                 delta_image=__delta_image,
                 delta_fourier=__delta_fourier,
                 anti_alias=True):
        """ Simulation class computes the optical properties of the ensemble of dipoles

            public attributes:
                ensemble (ensemble): Ensemble object containing all dipole objects
                lambda_0 (array): vacuum wavelengths for the simulation [µm]
                px_x (int): image resolution in x-direction
                px_y (int): image resolution in y-direction
                eps (float): permittivity of the surrounding
                mu (float): permeability of the surrounding
                na (float): normalized numerical aperture of the objective
                phi (float): azimuthal angle of the illumination [deg]
                theta (float): polar angle of the illumination [deg]
                dz (float): height of the integration volume [µm]
                delta_real (float): margin for the integration volume in the real space
                delta_fourier (float): margin for the integration volume in the fourier
                    space
                anti_alias (bool): ensure a sufficient grid resolution to avoid aliasing

            protected attributes:
                _K_x (array): momentum grid of the x-component in the fourier plane [1/µm]
                _K_y (array): momentum grid of the y-component in the fourier plane [1/µm]
                _X (array): spatial grid of the x-component in the image plane [µm]
                _Y (array): spatial grid of the y-component in the fourier plane [µm]
                _n (float): refractive index of the surrounding
                _k_0 (array): vacuum wavenumbers for the simulation [1/µm]
                _k (array): wavenumbers for the simulation [1/µm]
                _k_x (array): x-component of _k [1/µm]
                _k_y (array): y-component of _k [1/µm]
                _k_z (array): z-component of _k [1/µm]
                _size_image (tuple): tuple of the image plane size in x- and y-direction [µm]
                _size_fourier (tuple): tuple of the converted image plane size of the fourier plane
                    in x- and y-direction [µm]

            public methods:
                initialize: computes the transfer matrix and dipole moments
                transfer_matrix: generator that returns the matrix that connects the incident fields with the
                    dipole moments
                dipole_moment: returns an array of the x,y,z components of the dipole moment of each dipole for s- and
                    p-polarized incident electric fields
                reflection: return hyperspectral instances of the simulated reflection data in the image and fourier
                    plane

            protected methods:
                _alpha_inv: returns an array of inverse polarizabilities for each wavelength
                _kappa: returns the coupling matrix for each wavelength
                _radiation_angle (static): convert wavenumber grid to radiation angle grid
                _field_polarization (static): returns the field polarizations on the fourier grid
                _field_amplitude (static): returns the field amplitudes on the fourier grid
                _fourier_fields (static): returns the electric fields in the fourier plane
                _fourier_transform (static): transforms the electric fields in the fourier plane into the image plane

        """

        # check whether the input is of the right class
        if not isinstance(ensemble, Ensemble):
            raise ValueError('Input argument must be a ensemble object'
                             + ' but argument of type {0} was given' \
                             .format(type(ensemble)))

        # construct the public instance attributes
        self.ensemble = ensemble
        self.lambda_0 = np.array(lambda_0).reshape((-1,))
        self.px_x = px_x if px_x >= 2 else 2
        self.px_y = px_y if px_y >= 2 else 2
        self.eps = eps
        self.mu = mu
        self.na = na if (na >= 0) & (na <= 1) else 1
        self.phi = phi
        self.theta = theta
        self.dz = dz if dz <= 0 else 0
        self.delta_image = delta_image if delta_image >= 1.0 else 1.0
        self.delta_fourier = delta_fourier if delta_fourier >= 1.0 else 1.0

        # adapt the grid resolution if it doesn't comply with nyquist theorem
        if (self._size_fourier[0] < self._size_image[0]) & anti_alias:
            # save the current resolution
            px_x_old = self.px_x
            # update the resolution
            k_max = np.max(self._k)
            df = self.delta_fourier
            self.px_x = int(np.ceil(self._size_image[0] * k_max * df / np.pi))
            # print warning that grid resolution was changed
            message = "px_x was changed from {px_x_old} to {px_x}".format(px_x_old=px_x_old, px_x=self.px_x)
            warn(message, UserWarning)
        if (self._size_fourier[1] < self._size_image[1]) & anti_alias:
            # save the current resolution
            px_y_old = self.px_y
            # update the resolution
            k_max = np.max(self._k)
            df = self.delta_fourier
            self.px_y = int(np.ceil(self._size_image[1] * k_max * df / np.pi))
            # print warning that grid resolution was changed
            message = "px_y was changed from {px_y_old} to {px_y}".format(px_y_old=px_y_old, px_y=self.px_y)
            warn(message, UserWarning)

        # construct the protected instance attributes
        self._dipole_moment = None

    # getter for the spatial grid in the image plane in x-direction
    @property
    def _X(self):
        x = np.linspace(- self._size_image[0] / 2, self._size_image[0] / 2, self.px_x)
        y = np.linspace(- self._size_image[1] / 2, self._size_image[1] / 2, self.px_y)
        X = np.meshgrid(x, y)[0]

        return X

    # getter for the spatial grid in the image plane in y-direction
    @property
    def _Y(self):
        x = np.linspace(- self._size_image[0] / 2, self._size_image[0] / 2, self.px_x)
        y = np.linspace(- self._size_image[1] / 2, self._size_image[1] / 2, self.px_y)
        Y = np.meshgrid(x, y)[1]

        return Y

    # getter for the grid in the fourier plane
    @property
    def _K_x(self):
        # get the fourier plane coordinates
        k_x = self.delta_fourier * np.max(self._k) * np.linspace(-1, 1, self.px_x)
        k_y = self.delta_fourier * np.max(self._k) * np.linspace(-1, 1, self.px_y)
        K_x = np.meshgrid(k_x, k_y)[0]

        return K_x.reshape((-1, 1))

    # getter for the grid in the fourier plane
    @property
    def _K_y(self):
        # get the fourier plane coordinates
        k_x = self.delta_fourier * np.max(self._k) * np.linspace(-1, 1, self.px_x)
        k_y = self.delta_fourier * np.max(self._k) * np.linspace(-1, 1, self.px_y)
        K_y = np.meshgrid(k_x, k_y)[1]

        return K_y.reshape((-1, 1))

    # getter for the refractive index
    @property
    def _n(self):
        return np.sqrt(self.mu * self.eps)

    # getter for the incident vacuum wavenumber
    @property
    def _k_0(self):
        return 2 * np.pi / self.lambda_0

    # getter for the incident wavenumber
    @property
    def _k(self):
        return self._n * self._k_0

    # getter for the x-component of the incident wavevector
    @property
    def _k_x(self):
        # get the incident angles
        phi = np.deg2rad(self.phi)
        theta = np.deg2rad(self.theta)
        return self._k * np.sin(theta) * np.cos(phi)

    # getter for the y-component of the incident wavevector
    @property
    def _k_y(self):
        # get the incident angles
        phi = np.deg2rad(self.phi)
        theta = np.deg2rad(self.theta)
        return self._k * np.sin(theta) * np.sin(phi)

    # getter for the z-component of the incident wavevector
    @property
    def _k_z(self):
        # get the incident wavenumbers
        k = self._k
        k_x = self._k_x
        k_y = self._k_y

        return np.sqrt(k ** 2 - k_x ** 2 - k_y ** 2)

    # getter for the field size in the image plane
    @property
    def _size_image(self):
        # get the range of x and y positions
        rng_x = np.max(self.ensemble.x) - np.min(self.ensemble.x)
        rng_y = np.max(self.ensemble.y) - np.min(self.ensemble.y)
        # get the size of the image plane
        sz_x = (rng_x + 2) * self.delta_image  # add a 2µm margin
        sz_y = (rng_y + 2) * self.delta_image  # add a 2µm margin
        # add x and y component to tuple
        sz = (sz_x, sz_y)

        return sz

    # getter for the nyquist compliant image plane size from fourier plane size
    @property
    def _size_fourier(self):
        # get the nyquist compliant image plane size
        sz_x = self.px_x * np.pi / (np.max(np.real(self._k)) * self.delta_fourier)
        sz_y = self.px_y * np.pi / (np.max(np.real(self._k)) * self.delta_fourier)
        sz = (sz_x, sz_y)

        return sz

    # calculate the dipole moments for each wavelength and set the _dipole_moment attribute
    def initialize(self):
        # initialize the array with all dipole moments at each wavelength
        self._dipole_moment = np.zeros((3 * len(self.ensemble.dipoles), 2, len(self.lambda_0)), dtype=complex)

        for ii, lambda_0 in enumerate(self.lambda_0):
            self._dipole_moment[:, :, ii] = self.dipole_moment(lambda_0)

    # calculate the transfer matrix
    def transfer_matrix(self, lambda_0):
        # get the inverse polarizabilities and the coupling tensor
        transfer_matrix = self._alpha_inv(lambda_0) - self._kappa(lambda_0)

        return transfer_matrix

    # calculate the dipole moment of each dipole
    def dipole_moment(self, lambda_0):
        # get the positions of the dipoles
        x = self.ensemble.x
        y = self.ensemble.y
        z = self.ensemble.z
        # refractive index of the surrounding
        n = self._n
        # get the wavenumbers
        k = n * 2 * np.pi / lambda_0
        k_x = k * np.sin(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi))
        k_y = k * np.sin(np.deg2rad(self.theta)) * np.sin(np.deg2rad(self.phi))
        k_z = np.sqrt(k ** 2 - k_x ** 2 - k_y ** 2)
        # convert angles from degree to radiants
        phi = np.deg2rad(self.phi)
        theta = np.deg2rad(self.theta)

        # calculate the s- and p-polarized electric field amplitudes
        E_p = np.array([[np.cos(theta) * np.cos(phi)],
                        [np.cos(theta) * np.sin(phi)],
                        [-np.sin(theta)]])
        E_s = np.array([[-np.sin(phi)],
                        [np.cos(phi)],
                        [0]])

        # calculate the phase at the positions of the dipoles
        phase = np.exp(-1j * (k_x * x + k_y * y + k_z * z))
        # convert to row vector
        phase = phase.reshape(-1, 1)

        # get the incident field at the positions of each dipole
        E_p_inc = np.kron(phase, E_p)
        E_s_inc = np.kron(phase, E_s)
        E_inc = np.concatenate((E_p_inc, E_s_inc), axis=1) * 1e-6  # V/m to V/µm

        # get the transfer matrix at this wavelength
        tm = self.transfer_matrix(lambda_0)

        # calculate the dipole moments
        try:  # if transfer matrix has full rank
            p = np.linalg.solve(tm, E_inc * (epsilon_0 * n ** 2))
        except np.linalg.LinAlgError as err:  # if transfer matrix is rank deficient
            if 'Singular matrix' in str(err):
                p = np.linalg.lstsq(tm, E_inc * (epsilon_0 * n ** 2), rcond=None)[0]
            else:
                raise

        return p

    # calculate the reflected hyperspectral fields in the image and fourier plane
    def reflection(self):
        # get the wavelength to simulate
        lambda_0 = self.lambda_0
        # get wavenumbers
        k = self._k
        k_z = self._k_z
        # get simulation parameters
        phi_inc = np.deg2rad(self.phi)
        theta_inc = np.deg2rad(self.theta)
        na = self.na
        dz = self.dz
        mu = self.mu
        epsilon = self.eps
        # get the dipole positions
        x = self.ensemble.x
        y = self.ensemble.y
        z = self.ensemble.z
        # get the grid resolution
        px_x = self.px_x
        px_y = self.px_y
        # get the fourier grid coordinates
        K_x = self._K_x
        K_y = self._K_y
        sin_xi = K_x.reshape((px_x, px_y)) / np.max(k)
        sin_yi = K_y.reshape((px_x, px_y)) / np.max(k)
        # get the spatial grid of the image plane
        X = self._X
        Y = self._Y
        # calculate the integration volume
        dK_x = (np.max(K_x) - np.min(K_x))
        dK_y = (np.max(K_y) - np.min(K_y))
        integration_surface = dK_x * dK_y / self.delta_fourier ** 2
        # get the dipole moments
        P = self._dipole_moment

        # initialize the hyperspectral electric field arrays
        E_xx_fourier = np.zeros((px_x, px_y, len(self.lambda_0)), dtype=complex)
        E_xy_fourier = np.zeros((px_x, px_y, len(self.lambda_0)), dtype=complex)
        E_yx_fourier = np.zeros((px_x, px_y, len(self.lambda_0)), dtype=complex)
        E_yy_fourier = np.zeros((px_x, px_y, len(self.lambda_0)), dtype=complex)
        E_xx_image = np.zeros((px_x, px_y, len(self.lambda_0)), dtype=complex)
        E_xy_image = np.zeros((px_x, px_y, len(self.lambda_0)), dtype=complex)
        E_yx_image = np.zeros((px_x, px_y, len(self.lambda_0)), dtype=complex)
        E_yy_image = np.zeros((px_x, px_y, len(self.lambda_0)), dtype=complex)

        # initialize a matrix for the incident fields
        B_p = np.empty((px_x * px_y, 3 * x.size), dtype=complex)
        B_s = np.empty((px_x * px_y, 3 * x.size), dtype=complex)

        # convert momentum grid to grid of polar and azimuthal angle
        # cos_phi, sin_phi, cos_theta, sin_theta = Simulator._radiation_angle(np.max(k), K_x, K_y)
        # precompute constants
        # c_xx = (sin_phi ** 2 * (1 - cos_theta) + cos_theta)
        # c_yx = (- sin_phi * cos_phi) * (1 - cos_theta)
        # c_zx = (cos_phi * sin_theta)
        # c_xy = (- cos_phi * sin_phi * (1 - cos_theta))
        # c_yy = (cos_phi ** 2 * (1 - cos_theta) + cos_theta)
        # c_zy = (sin_phi * sin_theta)
        # c_xz = (- cos_phi * sin_theta)
        # c_yz = (- sin_phi * sin_theta)
        # c_zz = (cos_theta)

        # normalization constant to ensure equal power of fft (see parseval theorem)
        dx, dy = self._size_fourier
        fft_norm = 2 * np.pi / (dx * dy)

        for ii, value in enumerate(lambda_0):
            # get the absolute in plane momentum
            K_abs = np.sqrt(np.abs(K_x) ** 2 + np.abs(K_y) ** 2)
            # get the z-component of the momentum grid
            K_z = np.sqrt(k[ii] ** 2 - K_abs ** 2, dtype=complex)

            # calculate the field normalization factors
            normalization = 1j / (2 * EPSILON_0 * epsilon * integration_surface) \
                            * k[ii] ** 2 / k_z[ii]

            # get the field polarization vectors
            phase, E_px_0, E_py_0, E_pz_0, E_sx_0, E_sy_0 \
                = Simulator._field_polarization(x, y, z, K_x, K_y, K_z, K_abs, k[ii])

            # calculate field amplitudes of outgoing fields in the fourier plane
            A_pp, A_sp, A_ps, A_ss \
                = Simulator._field_amplitude(P[:, :, ii], phase, E_px_0, E_py_0, E_pz_0, E_sx_0, E_sy_0, B_p, B_s,
                                             normalization)

            # calculate complex outgoing fields in the fourier plane
            E_px, E_py, E_pz, E_sx, E_sy, E_sz \
                = Simulator._fourier_field(A_pp, A_sp, A_ps, A_ss, E_px_0, E_py_0, E_pz_0, E_sx_0, E_sy_0,
                                           K_z, dz)

            # consider polarization rotation caused by the objective
            # E_px = E_px * c_xx + E_py * c_yx + E_pz * c_zx
            # E_py = E_px * c_xy + E_py * c_yy + E_pz * c_zy
            # E_pz = E_px * c_xz + E_py * c_yz + E_pz * c_zz
            # E_sx = E_sx * c_xx + E_sy * c_yx + E_sz * c_zx
            # E_sy = E_sx * c_xy + E_sy * c_yy + E_sz * c_zy
            # E_sz = E_sx * c_xz + E_sy * c_yz + E_sz * c_zz

            # convert fields to xy-polarized fields
            E_xx_f = (E_px * np.cos(theta_inc) * np.cos(phi_inc) - E_sx * np.sin(phi_inc)).reshape((px_x, px_y))
            E_xy_f = (E_py * np.cos(theta_inc) * np.cos(phi_inc) - E_sy * np.sin(phi_inc)).reshape((px_x, px_y))
            E_yx_f = (E_px * np.cos(theta_inc) * np.sin(phi_inc) + E_sx * np.cos(phi_inc)).reshape((px_x, px_y))
            E_yy_f = (E_py * np.cos(theta_inc) * np.sin(phi_inc) + E_sy * np.cos(phi_inc)).reshape((px_x, px_y))

            # transform the fields from E(k_x, k_y) to E(sin_x, sin_y)
            sin_x = K_x / k[ii]
            sin_y = K_y / k[ii]
            cos_x = np.sqrt(1 - sin_x ** 2, dtype=complex)
            cos_y = np.sqrt(1 - sin_y ** 2, dtype=complex)
            transformation_scaling = k[ii] * np.sqrt(cos_x * cos_y).reshape((px_x, px_y)) # jacobian determinant
            sin_x = sin_x[:px_x]
            sin_y = sin_y[::px_x]
            E_xx_ft = RectBivariateSpline(sin_x, sin_y,
                                        np.real(E_xx_f * transformation_scaling)).ev(sin_xi, sin_yi) \
                       + 1j * RectBivariateSpline(sin_x, sin_y,
                                        np.imag(E_xx_f * transformation_scaling)).ev(sin_xi, sin_yi)
            E_xy_ft = RectBivariateSpline(sin_x, sin_y,
                                        np.real(E_xy_f * transformation_scaling)).ev(sin_xi, sin_yi) \
                       + 1j * RectBivariateSpline(sin_x, sin_y,
                                        np.imag(E_xy_f * transformation_scaling)).ev(sin_xi, sin_yi)
            E_yx_ft = RectBivariateSpline(sin_x, sin_y,
                                        np.real(E_yx_f * transformation_scaling)).ev(sin_xi, sin_yi) \
                       + 1j * RectBivariateSpline(sin_x, sin_y,
                                        np.imag(E_yx_f * transformation_scaling)).ev(sin_xi, sin_yi)
            E_yy_ft = RectBivariateSpline(sin_x, sin_y,
                                        np.real(E_yy_f * transformation_scaling)).ev(sin_xi, sin_yi) \
                       + 1j * RectBivariateSpline(sin_x, sin_y,
                                        np.imag(E_yy_f * transformation_scaling)).ev(sin_xi, sin_yi)

            # set the values outside of the NA to nan
            idx = (sin_xi ** 2 + sin_yi ** 2) > na ** 2
            E_xx_ft[idx] = np.nan
            E_xy_ft[idx] = np.nan
            E_yx_ft[idx] = np.nan
            E_yy_ft[idx] = np.nan

            # save the hyperspectral fields in the fourier plane
            E_xx_fourier[:, :, ii] = E_xx_ft.T
            E_xy_fourier[:, :, ii] = E_xy_ft.T
            E_yx_fourier[:, :, ii] = E_yx_ft.T
            E_yy_fourier[:, :, ii] = E_yy_ft.T

            # remove parts that are outside the given NA
            idx = (K_abs.reshape((px_x, px_y)) / np.real(k[ii])) > na
            E_xx_f[idx] = 0
            E_xy_f[idx] = 0
            E_yx_f[idx] = 0
            E_yy_f[idx] = 0

            # fourier transform fields to get the fields in the image plane
            E_xx_i = fft_norm * Simulator._fourier_transform(X, Y, K_x, K_y, px_x, px_y, E_xx_f)
            E_xy_i = fft_norm * Simulator._fourier_transform(X, Y, K_x, K_y, px_x, px_y, E_xy_f)
            E_yx_i = fft_norm * Simulator._fourier_transform(X, Y, K_x, K_y, px_x, px_y, E_yx_f)
            E_yy_i = fft_norm * Simulator._fourier_transform(X, Y, K_x, K_y, px_x, px_y, E_yy_f)

            # save the hyperspectral fields in the image plane
            E_xx_image[:, :, ii] = E_xx_i.reshape((px_x, px_y))
            E_xy_image[:, :, ii] = E_xy_i.reshape((px_x, px_y))
            E_yx_image[:, :, ii] = E_yx_i.reshape((px_x, px_y))
            E_yy_image[:, :, ii] = E_yy_i.reshape((px_x, px_y))

        # instantiate the hyperspectral objects
        fourier = Hyperspectral(sin_xi, sin_yi, lambda_0,
                                E_xx_fourier,
                                E_xy_fourier,
                                E_yx_fourier,
                                E_yy_fourier)
        image = Hyperspectral(1e-6 * X, 1e-6 * Y, lambda_0,
                              E_xx_image * 1e6,
                              E_xy_image * 1e6,
                              E_yx_image * 1e6,
                              E_yy_image * 1e6)

        return fourier, image

    # get the inverse polarizability tensor of all dipoles
    def _alpha_inv(self, lambda_0):
        # get all dipoles
        dipoles = self.ensemble.dipoles

        # initialize array with all polarizabilties at the current wavenumber
        alpha_inv = np.zeros((3 * len(dipoles), 3 * len(dipoles)), dtype=complex)

        # loop through each dipole
        for ii, dipole in enumerate(dipoles):
            # resample the polarizabilities
            alpha_xx = dipole.alpha_xx(lambda_0)
            alpha_yy = dipole.alpha_yy(lambda_0)
            alpha_zz = dipole.alpha_zz(lambda_0)

            # get the normal matrix with the dipole orientation
            normal_matrix = np.concatenate((dipole.n1, dipole.n2, dipole.n3), axis=1)

            # create inverse polarizability tensor
            alpha_inv_dipole = np.diag((1 / alpha_xx, 1 / alpha_yy, 1 / alpha_zz))
            # rotate the tensor along its orientation
            alpha_inv_dipole = normal_matrix @ alpha_inv_dipole @ normal_matrix.T

            # add the inverse polarizability matrix to the output
            alpha_inv[3 * ii:3 * (ii + 1), 3 * ii:3 * (ii + 1)] = alpha_inv_dipole

        return alpha_inv

    # get the coupling matrix
    def _kappa(self, lambda_0):
        # get the refractive index of the surrounding
        n = self._n
        # get wavenumber
        k = n * 2 * np.pi / lambda_0

        # get the dipole positions
        x = self.ensemble.x
        y = self.ensemble.y
        z = self.ensemble.z
        # calculate auxiliary matrices
        r_row_1 = np.kron(x, np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])) \
                  + np.kron(y, np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])) \
                  + np.kron(z, np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]))
        r_row_2 = np.kron(x, np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])) \
                  + np.kron(y, np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])) \
                  + np.kron(z, np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]))
        r_col_1 = np.kron(x.T, np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])) \
                  + np.kron(y.T, np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])) \
                  + np.kron(z.T, np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]))
        r_col_2 = np.kron(x.T, np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])) \
                  + np.kron(y.T, np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])) \
                  + np.kron(z.T, np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]))
        # get the outer product
        RR = (np.matlib.repmat(r_col_1, 1, x.shape[1]) - np.matlib.repmat(r_row_1, x.shape[1], 1)) \
             * (np.matlib.repmat(r_col_2, 1, x.shape[1]) - np.matlib.repmat(r_row_2, x.shape[1], 1))

        # get the euclidean distance of each dipole
        dx = distance.cdist(x.T, x.T, 'euclidean')
        dy = distance.cdist(y.T, y.T, 'euclidean')
        dz = distance.cdist(z.T, z.T, 'euclidean')
        # get the absolute distances
        R = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        # construct auxiliary matrices
        s_mat = np.kron(R != 0, np.ones((3, 3)))
        id_mat = np.matlib.repmat(np.eye(3), len(dx), len(dx))
        # bring the matrix to the correct form
        R = np.kron(R, np.ones((3, 3)))
        # replace zero entries to avoid zero division
        R[s_mat == 0] = 1

        # normalize the array
        RR = RR / R ** 2

        # calculate the spherical wave
        E_sphere = np.exp(1j * k * R) / (4 * np.pi * R)
        # calculate near, intermediate and far field
        E_near = (-id_mat + 3 * RR) / (k * R) ** 2
        E_inter = 1j * (id_mat - 3 * RR) / (k * R)
        E_far = (id_mat - RR)

        # calculate the coupling matrix
        kappa = s_mat * k ** 2 * (E_near + E_inter + E_far) * E_sphere

        return kappa

    # convert momentum grid to polar and azimuthal angles
    @staticmethod
    def _radiation_angle(k, K_x, K_y):
        # get the absolute in plane momentum
        K_abs = np.sqrt(np.abs(K_x) ** 2 + np.abs(K_y) ** 2)
        # get the z-component of the momentum grid
        K_z = np.sqrt(np.abs(k) ** 2 - K_abs ** 2, dtype=complex)

        # suppress zero division warnings
        np.seterr(divide='ignore')
        # calculate the inverse absolute momentum grid
        K_abs_inv = np.where(np.isnan(1 / K_abs), 0, 1 / K_abs)
        K_abs_inv = np.where(np.isinf(K_abs_inv), 1e12, K_abs_inv)
        # activate zero division warnings
        np.seterr(divide='warn')

        # calculate the angles
        cos_phi = K_x * K_abs_inv
        sin_phi = K_y * K_abs_inv
        cos_theta = K_z / k
        sin_theta = K_abs / k

        return cos_phi, sin_phi, cos_theta, sin_theta

    # calculate incident fields
    @staticmethod
    def _field_polarization(x, y, z, K_x, K_y, K_z, K_abs, k):
        # change the sign of bound and backward propagating waves
        idx = (np.abs(np.imag(K_z)) > np.finfo(float).eps) & (np.imag(K_z) < 0) \
              | (np.abs(np.imag(K_z)) <= np.finfo(float).eps) & (np.real(K_z) < 0)
        K_z[idx] = - K_z[idx]
        # get indices of small absolute or z-component momentum
        idx_abs = K_abs < 1e-10
        idx_z = np.abs(K_z) < 1e-10

        # suppress zero division warnings
        np.seterr(divide='ignore')
        # calculate the inverse absolute momentum grid
        K_abs_inv = np.where(np.isnan(1 / K_abs), 0, 1 / K_abs)
        K_abs_inv = np.where(np.isinf(K_abs_inv), 1e12, K_abs_inv)
        # activate zero division warnings
        np.seterr(divide='warn')

        # field amplitudes for p-polarized incidence from top
        E_px_0 = - K_abs_inv / k * K_z * K_x
        E_px_0[idx_abs] = -1
        E_px_0[idx_z] = 0
        E_py_0 = - K_abs_inv / k * K_z * K_y
        E_py_0[idx_abs] = 0
        E_py_0[idx_z] = 0
        E_pz_0 = -1 / k * K_abs
        E_pz_0[idx_abs] = 0
        # field amplitudes for s-polarized incidence from top
        E_sx_0 = K_abs_inv * K_y
        E_sx_0[idx_abs] = 0
        E_sy_0 = - K_abs_inv * K_x
        E_sy_0[idx_abs] = -1
        # get the phase in the xy-plane
        phase = np.exp(-1j * (K_x * x + K_y * y + K_z * z))

        return phase, E_px_0, E_py_0, E_pz_0, E_sx_0, E_sy_0

    # calculate the field amplitudes in the fourier plane
    @staticmethod
    def _field_amplitude(P, phase, E_px_0, E_py_0, E_pz_0, E_sx_0, E_sy_0, B_p, B_s, normalization):
        # create an auxiliary matrix to repeatedly concatenate the incident fields
        mat_aux = np.ones((1, phase.shape[1]))

        # calculate the incident fields
        B_p[:, 0::3] = phase * (E_px_0 @ mat_aux)
        B_p[:, 1::3] = phase * (E_py_0 @ mat_aux)
        B_p[:, 2::3] = phase * (E_pz_0 @ mat_aux)
        B_s[:, 0::3] = phase * (E_sx_0 @ mat_aux)
        B_s[:, 1::3] = phase * (E_sy_0 @ mat_aux)
        B_s[:, 2::3] = np.zeros(phase.shape)

        # calculate the outgoing field amplitudes in the fourier plane
        A_p = normalization * (B_p @ P)
        A_s = normalization * (B_s @ P)

        # get the different amplitude components
        A_pp = A_p[:, :1]
        A_sp = A_p[:, 1:]
        A_ps = A_s[:, :1]
        A_ss = A_s[:, 1:]

        return A_pp, A_sp, A_ps, A_ss

    # calculate the outgoing complex fields in the fourier plane
    @staticmethod
    def _fourier_field(A_pp, A_sp, A_ps, A_ss, E_px_0, E_py_0, E_pz_0, E_sx_0, E_sy_0, K_z, dz):
        # calculate the field amplitudes
        E_px = (A_pp * E_px_0 + A_ps * E_sx_0) * np.exp(- 2j * K_z * dz)
        E_py = (A_pp * E_py_0 + A_ps * E_sy_0) * np.exp(- 2j * K_z * dz)
        E_pz = (A_pp * E_pz_0) * np.exp(- 2j * K_z * dz)
        E_sx = (A_sp * E_px_0 + A_ss * E_sx_0) * np.exp(- 2j * K_z * dz)
        E_sy = (A_sp * E_py_0 + A_ss * E_sy_0) * np.exp(- 2j * K_z * dz)
        E_sz = (A_sp * E_pz_0) * np.exp(- 2j * K_z * dz)

        return E_px, E_py, E_pz, E_sx, E_sy, E_sz

    # calculate the fourier transformation from fourier to image plane
    @staticmethod
    def _fourier_transform(X, Y, K_x, K_y, px_x, px_y, E):
        # reshape the arrays
        x = X[0, :].reshape((-1, 1))
        y = Y[:, 0].reshape((-1, 1))
        k_x = K_x[:px_x].T
        k_y = K_y[::px_x].T
        E = E.reshape(px_x, px_y)

        # get the FFT signal
        E_FFT = (np.exp(1j * x @ k_x)
                 @ (np.exp(1j * y @ k_y)
                    @ E).T
                 ).T

        return E_FFT
