import numpy as np
from sklearn.neighbors import KDTree


def tpcf_radial(coordinates, n_bins=100, r_lim=None, factor=50, seed=42):
    """ Function to calculate the radial two point correlation function (tpcf) of a n-dimensional
    coordinate grid according to the landy szalay estimator
                arguments:
                    coordinates (array): n-dimensional coordinate grid with shape n_sample x n_dim where n_sample is the
                        number of samples and n_dim is the number of dimensions
                    n_bins (int): number of radial bins to calculate the tpcf for
                    r_lim (list, tuple or None): list or tuple with lower and upper limits for the radial vector.
                        If None is specified, the range is automatically determined
                    factor (int): multiplication factor that determines how much more (factor > 1) or less (factor < 1)
                        random samples are given in comparison to the original coordinate samples
                    seed (int): seed for the random number generator that is used to generate the random comparison
                        coordinates

                returns:
                    r (array): radial vector of length n_bins that determines the radial position of each value in tpcf
                    tpcf (array): vector of length n_bins with computed values of the tpcf for each radial position in r
    """

    # set the seed of the random number generator
    np.random.seed(seed=seed)

    # get the number of dimensions and the number of samples per dimension
    n_sample = coordinates.shape[0]
    n_dim = coordinates.shape[1]

    # shift the coordinates to the origin
    coordinates = coordinates - np.min(coordinates, axis=0)
    # get the maximum distance along each dimension of the coordinate grid
    max_range = (np.max(coordinates, axis=0) - np.min(coordinates, axis=0)).reshape((1, -1))
    max_dist = np.sqrt(np.sum(max_range ** 2))

    # create the randomized coordinates
    n_sample_random = round(n_sample * factor)
    max_range = np.tile(max_range, (n_sample_random, 1))
    random_coordinates = np.random.random((n_sample_random, n_dim)) * max_range

    # create the radial binning vector
    if r_lim is None:
        r = np.linspace(0, max_dist * (1 + 1 / (n_bins - 1)), n_bins + 1)
    else:
        r = np.linspace(r_lim[0], r_lim[1] + (r_lim[1] - r_lim[0]) / (n_bins - 1), n_bins + 1)

    # fast two-point correlation functions
    KDT_D = KDTree(coordinates)
    KDT_R = KDTree(random_coordinates)

    # get the number of pairs in less or equal distance to r[i]
    counts_DD = KDT_D.two_point_correlation(coordinates, r)
    counts_RR = KDT_R.two_point_correlation(random_coordinates, r)
    counts_DR = KDT_R.two_point_correlation(coordinates, r)

    # get the number of pairs in each bin
    DD = np.diff(counts_DD)
    RR = np.diff(counts_RR)
    DR = np.diff(counts_DR)

    # check for zero in the denominator
    RR_zero = (RR == 0)
    RR[RR_zero] = 1

    # compute the two points correlation function (tpcf) according to the landy szalay estimator
    tpcf = (factor ** 2 * DD - 2 * factor * DR + RR) / RR

    # replace entries without pairs with nan
    tpcf[RR_zero] = np.nan

    return r[:-1], tpcf


def tpcf_cartesian(coordinates, n_bins=100, cart_lim=None, factor=50, seed=42):
    """ Function to calculate the cartesian two point correlation function (tpcf) of a n-dimensional
    coordinate grid according to the landy szalay estimator
                arguments:
                    coordinates (array): n-dimensional coordinate grid with shape n_sample x n_dim where n_sample is the
                        number of samples and n_dim is the number of dimensions
                    n_bins (int): number of cartesian bins to calculate the tpcf for
                    cart_lim (array): array of size 2 x n_dim that determines the lower and upper limit of each
                        dimension. If None is specified, the range is automatically determined
                    factor (int): multiplication factor that determines how much more (factor > 1) or less (factor < 1)
                        random samples are given in comparison to the original coordinate samples
                    seed (int): seed for the random number generator that is used to generate the random comparison
                        coordinates

                returns:
                    cartesian (array): array of shape n_bins x n_dim with each column being the coordinate vector
                        of a certain dimension
                    tpcf (array): array of shape n_bins x n_bins with the computed values of the tpcf for each cartesian
                        position
    """

    # set the seed of the random number generator
    np.random.seed(seed=seed)

    # get the number of dimensions and the number of samples per dimension
    n_sample = coordinates.shape[0]
    n_dim = coordinates.shape[1]

    # get the coordinate border points
    coordinate_min = np.min(coordinates, axis=0)
    coordinate_max = np.max(coordinates, axis=0)
    coordinate_range = coordinate_max - coordinate_min

    # get the maximum distance along each dimension of the coordinate grid
    if cart_lim is None:
        # get the limits of the cartesian grid
        cart_lim = np.zeros((2, n_dim))
        cart_lim[0, :] = coordinate_min - coordinate_max
        cart_lim[1, :] = coordinate_max - coordinate_min

    # create the randomized coordinates
    n_sample_random = round(n_sample * factor)
    random_coordinates = (np.random.random((n_sample_random, n_dim)) - 0.5) * coordinate_range

    # initialize the number of pairs counters
    DD = np.zeros((n_bins, n_bins))
    DR = np.zeros((n_bins, n_bins))
    RR = np.zeros((n_bins, n_bins))

    # loop through all coordinates
    for origin in coordinates:
        delta_coord_dd = coordinates - origin
        delta_coord_dd = delta_coord_dd[np.any(delta_coord_dd!=0, axis=1), :]
        hist_dd, cartesian = np.histogramdd(delta_coord_dd,
                        bins=n_bins, range=tuple(zip(cart_lim[0, :], cart_lim[1, :])))
        DD = DD + hist_dd

        delta_coord_dr = random_coordinates - origin
        hist_dr = np.histogramdd(delta_coord_dr,
                        bins=n_bins, range=tuple(zip(cart_lim[0, :], cart_lim[1, :])))[0]
        DR = DR + hist_dr

    # loop through all random coordinates
    for origin in random_coordinates:
        delta_coord_rr = random_coordinates - origin
        delta_coord_rr = delta_coord_rr[np.any(delta_coord_rr!=0, axis=1), :]
        hist_rr = np.histogramdd(delta_coord_rr,
                        bins=n_bins, range=tuple(zip(cart_lim[0, :], cart_lim[1, :])))[0]
        RR = RR + hist_rr

    # check for zero in the denominator
    RR_zero = (RR == 0)
    RR[RR_zero] = 1

    # compute the two point correlation function (tpcf) according to the landy szalay estimator
    tpcf = (factor ** 2 * DD - 2 * factor * DR + RR) / RR

    # replace entries without pairs with nan
    tpcf[RR_zero] = np.nan

    # reshape the cartesian coordinate list to array
    cartesian = np.array(cartesian).T
    cartesian = cartesian[:-1, :]

    return cartesian, tpcf


def structure_factor(coordinates, n_bins=100, q_lim=[- 2 * np.pi / 0.4, 2 * np.pi / 0.4]):
    """ Function to calculate the structure factor of a structure with a single type of particles
                    arguments:
                        coordinates (array): n-dimensional coordinate grid with shape n_sample x n_dim where n_sample is
                            the number of samples and n_dim is the number of dimensions
                        n_bins (int): number of reciprocal grid points per dimension at which the structure factor is
                            evaluated
                        q_lim (list): list that determines the lower and upper limit of the reciprocal space

                    returns:
                        q (array): array of shape n_bins x n_dim with each column being the coordinate vector
                            of a certain dimension
                        sf_tot (array): array of shape n_bins x n_bins with the computed values of the
                            structure factor for each reciprocal coordinate point
        """

    # get the number of sample points and dimensions
    n_sample = coordinates.shape[0]
    n_dim = coordinates.shape[1]

    # convert the number of bins to a list
    if type(n_bins) == int:
        n_bins = [n_bins for _ in range(n_dim)]

    # create the reciprocal coordinates
    q_x = np.linspace(q_lim[0], q_lim[1], n_bins[0])
    q_y = np.linspace(q_lim[0], q_lim[1], n_bins[1])
    q = [q_x, q_y]
    Q_x, Q_y = np.meshgrid(q_x, q_y)

    # initialize arrays of the structure factor components
    sf_cos = np.zeros((n_bins[0], n_bins[1]))
    sf_sin = np.zeros((n_bins[0], n_bins[1]))

    # loop through each individual structure
    for coordinate in coordinates:
        # separate the x- and y-coordinate
        x_i = coordinate[0]
        y_i = coordinate[1]

        # calculate the structure factor components
        sf_cos += np.cos(Q_x * x_i + Q_y * y_i)
        sf_sin += np.sin(Q_x * x_i + Q_y * y_i)

    # compute the total structure factor
    sf_tot = np.abs(sf_cos + 1j * sf_sin) ** 2 / n_sample

    return q, sf_tot



