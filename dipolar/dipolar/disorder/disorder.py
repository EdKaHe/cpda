import numpy as np
from dipolar.disorder.distributions import uniform1d, uniform2d, normal1d, normal2d, triangular1d, triangular2d


class Disorder:
    def __init__(self, n_x, n_y, p_x, p_y):
        """ Disorder class that helps to generate disordered positions, rotations and dimensions of dipoles

        public attributes:
            n_x (int): number of dipoles in x-direction
            n_y (int): number of dipoles in y-direction
            p_x (float): periodicity of dipoles in x-direction [µm]
            p_y (float): periodicity of dipoles in y-direction [µm]
            X, Y (array): 2d meshgrid of positions in x- and y-direction of each dipole

        public methods:
            positional: generates 2d grids of disordered x- and y-positions of each dipole  [µm]
            rotational: generates a 2d grid of disordered rotation angles of each dipole [deg]
            dimensional: generates a 2d grid of disordered relative dimension deviations of each dipole

        protected methods:
            _coordinate_grid: generates a periodic coordinate meshgrid in x- and y-direction for the given attributes

        """

        # construct the instance attributes
        self.n_x = n_x
        self.n_y = n_y
        self.p_x = p_x
        self.p_y = p_y

        # initialize a grid
        self.__X, self.__Y = self._coordinate_grid()
        self.X = self.__X.copy()
        self.Y = self.__Y.copy()

    def positional(self, ds=0, cl=0, seed=None, ddist="uniform", cdist="normal"):
        # get the attributes
        n_x = self.n_x
        n_y = self.n_y
        p_x = self.p_x
        p_y = self.p_y
        # get the grid coordinates
        X_g = self.__X.copy()
        Y_g = self.__Y.copy()

        # return grid if disorder strength is zero
        if ds == 0:
            return X_g, Y_g

        # set the seed of the random number generator
        np.random.seed(seed=seed)

        # generate completely random positions if disorder strength is not finite
        if np.isinf(float(ds)):
            X_d = (n_x - 1) * p_x * (np.random.rand(n_x, n_y) - 0.5)
            Y_d = (n_y - 1) * p_y * (np.random.rand(n_x, n_y) - 0.5)

            return X_d, Y_d

        # get the distribution of the disorder
        if ddist == "uniform":
            pdf = uniform2d
        elif ddist == "normal":
            pdf = normal2d
        elif ddist == "triangular":
            pdf = triangular2d
        else:
            raise ValueError('ddist must be "uniform", "normal" or "triangular"')

        # generate random shifts
        dX, dY = pdf(n_x, n_y)
        dX *= ds * p_x
        dY *= ds * p_y

        # initialize disordered grid
        X_d = X_g + dX
        Y_d = Y_g + dY

        # introduce correlation
        if cl > 0:
            # get the indices from inner to outer dipoles
            squared_distance = (X_g - X_g[0]) ** 2 + (Y_g - Y_g[0]) ** 2
            dipole_indices_0 = np.unravel_index(np.argsort(squared_distance, axis=None), squared_distance.shape)
            dipole_indices = dipole_indices_0

            # iterate through each dipole
            for ii in range(squared_distance.size):
                # get the index of the current dipole
                idx = (dipole_indices_0[0][ii], dipole_indices_0[1][ii])

                # calculate the correlation weights
                if cdist == "uniform":
                    scaled_distance = np.sqrt((X_d - X_d[idx]) ** 2 / (0.5 * p_x * cl) ** 2 \
                                              + (Y_d - Y_d[idx]) ** 2 / (0.5 * p_y * cl) ** 2)
                    weight = scaled_distance <= 1
                elif cdist == "normal":
                    mu = (X_d[idx], Y_d[idx])
                    FWHM = (cl * p_x, cl * p_y)
                    FWHM_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
                    sigma = (FWHM[0] * FWHM_to_sigma, FWHM[1] * FWHM_to_sigma)
                    weight = np.exp(- (X_d - mu[0]) ** 2 / (2 * sigma[0] ** 2)) \
                             * np.exp(- (Y_d - mu[1]) ** 2 / (2 * sigma[1] ** 2))
                elif cdist == "triangular":
                    scaled_distance = np.sqrt((X_d - X_d[idx]) ** 2 / (p_x * cl) ** 2 \
                                              + (Y_d - Y_d[idx]) ** 2 / (p_y * cl) ** 2)
                    weight = 1 - scaled_distance
                    weight[scaled_distance > 1] = 0
                else:
                    raise ValueError('cdist must be "uniform", "normal" or "triangular"')

                # shift all dipoles by the weighted displacement of the current dipole
                weight[idx] = 0
                X_d[dipole_indices] = X_d[dipole_indices] + weight[dipole_indices] * dX[idx]
                Y_d[dipole_indices] = Y_d[dipole_indices] + weight[dipole_indices] * dY[idx]

        # add the disordered grid to the attributes
        self.X = X_d
        self.Y = Y_d

        return self.X, self.Y

    def rotational(self, ds=0, cl=0, seed=None, ddist="uniform", cdist="normal"):
        # get the attributes
        n_x = self.n_x
        n_y = self.n_y
        p_x = self.p_x
        p_y = self.p_y
        # get the grid coordinates
        X = self.X
        Y = self.Y

        # initialize the rotation grid
        Phi_0 = np.zeros((n_x, n_y))

        # return unrotated grid if disorder strength is zero
        if ds == 0:
            return Phi_0

        # set the seed of the random number generator
        np.random.seed(seed=seed)

        # generate completely random angles if disorder strength is not finite
        if np.isinf(ds) or ds == "inf":
            Phi = 360 * (np.random.rand(n_x, n_y) - 0.5)

            return Phi

        # get the distribution of the disorder
        if ddist == "uniform":
            pdf = uniform1d
        elif ddist == "normal":
            pdf = normal1d
        elif ddist == "triangular":
            pdf = triangular1d
        else:
            raise ValueError('ddist must be "uniform", "normal" or "triangular"')

        # generate random shifts
        Phi = Phi_0
        dPhi = 360 * ds * pdf(n_x, n_y)

        # introduce correlation
        if cl > 0:
            # get the indices from inner to outer dipoles
            squared_distance = (X - X[0]) ** 2 + (Y - Y[0]) ** 2
            dipole_indices_0 = np.unravel_index(np.argsort(squared_distance, axis=None), squared_distance.shape)
            dipole_indices = dipole_indices_0

            # iterate through each dipole
            for ii in range(squared_distance.size):
                # get the index of the current dipole
                idx = (dipole_indices_0[0][ii], dipole_indices_0[1][ii])

                # calculate the correlation weights
                if cdist == "uniform":
                    scaled_distance = np.sqrt((X - X[idx]) ** 2 / (0.5 * p_x * cl) ** 2 \
                                              + (Y - Y[idx]) ** 2 / (0.5 * p_y * cl) ** 2)
                    weight = scaled_distance <= 1
                elif cdist == "normal":
                    mu = (X[idx], Y[idx])
                    FWHM = (cl * p_x, cl * p_y)
                    FWHM_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
                    sigma = (FWHM[0] * FWHM_to_sigma, FWHM[1] * FWHM_to_sigma)
                    weight = np.exp(- (X - mu[0]) ** 2 / (2 * sigma[0] ** 2)) \
                             * np.exp(- (Y - mu[1]) ** 2 / (2 * sigma[1] ** 2))
                elif cdist == "triangular":
                    scaled_distance = np.sqrt((X - X[idx]) ** 2 / (p_x * cl) ** 2 \
                                              + (Y - Y[idx]) ** 2 / (p_y * cl) ** 2)
                    weight = 1 - scaled_distance
                    weight[scaled_distance > 1] = 0
                else:
                    raise ValueError('cdist must be "uniform", "normal" or "triangular"')

                # rotate all dipoles by the weighted rotation of the current dipole
                Phi[dipole_indices] = Phi[dipole_indices] + weight[dipole_indices] * dPhi[idx]

        else:
            # get the uncorrelated disorder
            Phi = Phi + dPhi

        # add the disordered grid to the attributes
        self.Phi = Phi

        return self.Phi

    def dimensional(self, ds=0, cl=0, seed=None, ddist="uniform", cdist="normal"):
        # get the attributes
        n_x = self.n_x
        n_y = self.n_y
        p_x = self.p_x
        p_y = self.p_y
        # get the grid coordinates
        X = self.X
        Y = self.Y

        # initialize the dimension grid
        dS_0 = np.zeros((n_x, n_y))

        # return unchanged dimension grid if disorder strength is zero
        if ds == 0:
            return dS_0

        # set the seed of the random number generator
        np.random.seed(seed=seed)

        # generate completely random dimensions if disorder strength is not finite
        if np.isinf(ds) or ds == "inf":
            dS = (np.random.rand(n_x, n_y) - 0.5)

            return dS

        # get the distribution of the disorder
        if ddist == "uniform":
            pdf = uniform1d
        elif ddist == "normal":
            pdf = normal1d
        elif ddist == "triangular":
            pdf = triangular1d
        else:
            raise ValueError('ddist must be "uniform", "normal" or "triangular"')

        # generate random dimensions
        dS = dS_0
        ddS = ds * pdf(n_x, n_y)

        # introduce correlation
        if cl > 0:
            # get the indices from inner to outer dipoles
            squared_distance = (X - X[0]) ** 2 + (Y - Y[0]) ** 2
            dipole_indices_0 = np.unravel_index(np.argsort(squared_distance, axis=None), squared_distance.shape)
            dipole_indices = dipole_indices_0

            # iterate through each dipole
            for ii in range(squared_distance.size):
                # get the index of the current dipole
                idx = (dipole_indices_0[0][ii], dipole_indices_0[1][ii])

                # calculate the correlation weights
                if cdist == "uniform":
                    scaled_distance = np.sqrt((X - X[idx]) ** 2 / (0.5 * p_x * cl) ** 2 \
                                              + (Y - Y[idx]) ** 2 / (0.5 * p_y * cl) ** 2)
                    weight = scaled_distance <= 1
                elif cdist == "normal":
                    mu = (X[idx], Y[idx])
                    FWHM = (cl * p_x, cl * p_y)
                    FWHM_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
                    sigma = (FWHM[0] * FWHM_to_sigma, FWHM[1] * FWHM_to_sigma)
                    weight = np.exp(- (X - mu[0]) ** 2 / (2 * sigma[0] ** 2)) \
                             * np.exp(- (Y - mu[1]) ** 2 / (2 * sigma[1] ** 2))
                elif cdist == "triangular":
                    scaled_distance = np.sqrt((X - X[idx]) ** 2 / (p_x * cl) ** 2 \
                                              + (Y - Y[idx]) ** 2 / (p_y * cl) ** 2)
                    weight = 1 - scaled_distance
                    weight[scaled_distance > 1] = 0
                else:
                    raise ValueError('cdist must be "uniform", "normal" or "triangular"')

                # change dimensions of all dipoles by the weighted dimension change of the current dipole
                dS[dipole_indices] = dS[dipole_indices] + weight[dipole_indices] * ddS[idx]

        else:
            # get the uncorrelated disorder
            dS = dS + ddS

        # add the disordered dimension to the attributes
        self.dS = dS

        return self.dS

    def _coordinate_grid(self):
        # get the attributes
        n_x = self.n_x
        n_y = self.n_y
        p_x = self.p_x
        p_y = self.p_y

        # create a periodic meshgrid
        x = np.linspace(- p_x * (n_x - 1) / 2, p_x * (n_x - 1) / 2, n_x)
        y = np.linspace(- p_y * (n_y - 1) / 2, p_y * (n_y - 1) / 2, n_y)
        X, Y = np.meshgrid(x, y)

        return X.T, Y.T
