import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from joblib import Parallel, delayed
import sys

# define constants
FLOAT_EPSILON = sys.float_info.epsilon


def retrieve_image(fourier_amp, support=None, beta=0.8, p=0, n_iter=10000, n_init=1, n_sigma=10, method="oss",
                   parallel=True, seed=None):
    """ Function that retrieves the phase of an amplitude only diffraction pattern and returns the
        corresponding image plane
        arguments:
            fourier_amp (array): array of the known amplitudes in the fourier plane
            support (array): array of the same size as fourier_amp that provides the support region
                in the image plane
            beta (float): float that specifies the step width in the update step
            n_iter (integer): number of iterations after which the algorithm terminates
            n_init (integer): number of random phase initializations
            method (str): the method that is used to retrieve the phase ("hio" for Fienup's hybrid input output
                algorithm or "oss" for oversampling smoothness algorithm)
            parallel (bool): true if phase initializations shall be executed in parallel
            seed (integer): random seed for the random number generator

        returns:
            image (array): recovered and complex valued image plane
    """

    if method not in ["oss", "hio"]:
        raise ValueError("'{0}' is an invalid input argument. Method must be 'oss' or 'hio'.".format(method))

    # make shallow copy of the fourier amplitudes
    fourier_amp = np.copy(fourier_amp)
    # normalize the fourier amplitudes
    fourier_amp /= np.max(fourier_amp)
    # get the number of pixels
    n_x, n_y = fourier_amp.shape
    # create artificial reciprocal coordinates
    q_x = np.linspace(-1, 1, n_x)
    q_y = np.linspace(-1, 1, n_y)
    Q_x, Q_y = np.meshgrid(q_x, q_y)

    # ensure that integers are of correct type
    n_iter = int(n_iter)
    n_init = int(n_init)

    # set the seed of the random number generator
    np.random.seed(seed=seed)

    # create initial guess for the phase in the fourier plane
    phase_init = np.random.random((n_x, n_y, n_init))

    # create the support mask if not specified
    if support is None:
        support = np.ones(fourier_amp.shape, dtype=bool)
    else:
        support = support.astype(bool)

    # initialize the standard deviations for the oss step
    if method == "oss":
        n_px = np.max([n_x, n_y])
        sigma = np.linspace(n_px, 1, n_sigma)
        if n_iter // n_sigma > 0:
            sigma = np.outer(sigma, np.ones((int(np.ceil(n_iter/n_sigma))))).flatten()
        else:
            sigma = np.outer(sigma, np.ones((1))).flatten()
    else:
        sigma = None

    # parallelize the algorithm initializations
    if parallel:
        n_jobs = -1
    else:
        n_jobs = 1
    image = Parallel(n_jobs=n_jobs)(
                delayed(
                    lambda x: retrieve_image_parallel(phase_init[:, :, x], fourier_amp, support, beta, p, n_iter, method,
                                                      sigma, Q_x, Q_y)
                        )(ii) for ii in range(n_init))

    # convert the list of 2d arrays to 3d array
    image = np.dstack(image)

    # compute the fourier plane amplitudes of the recovered image
    fourier_rec = np.abs(fftshift(fft2(image, axes=(0, 1)), axes=(0, 1)))
    # normalize the fourier planes
    fourier_rec /= np.max(fourier_rec, axis=(0, 1), keepdims=True)

    # calculate the errors
    error = np.sum(
        np.abs(
            np.log(fourier_rec + FLOAT_EPSILON) - np.log(fourier_amp[:, :, None] + FLOAT_EPSILON)
        ) ** 2,
        axis=(0, 1)
    )
    # get the index of the least error
    optimum_index = np.argmin(error)

    # normalize the image
    image = image[:, :, optimum_index]
    image /= np.max(image)

    return image


def retrieve_image_parallel(phase_init, fourier_amp, support, beta, p, n_iter, method, sigma, Q_x, Q_y):
    """ Helper function to parallelize the phase initializations of the image retrieval
        """

    # initialize initial guess for the complex fourier plane
    fourier_update = fourier_amp * np.exp(1j * 2 * np.pi * phase_init)
    # initialize previous image state
    image_old = None

    # iteratively optimize the phase
    for jj in range(n_iter):
        # obtain the image via inverse fourier transform
        image = np.real(ifft2(ifftshift(fourier_update)))

        # get the previous image state
        if image_old is None:
            image_old = image

        # update the last image state
        image_new = image

        # get elements that violate the image plane constraints
        no_support = ~support
        constraints = (image < np.percentile(image[image > 0], p * 100)) & support

        # updates for elements that violate image plane constraints
        image[no_support | constraints] = image_old[no_support | constraints] - beta * image[no_support | constraints]

        if method == "oss":
            # fourier transform the updated image
            fourier_guess = fftshift(fft2(image))
            # create a gaussian for the oversampling smoothness step
            gaussian = np.exp(- 0.5 * (Q_x ** 2 + Q_y ** 2) / sigma[jj] ** 2)
            gaussian /= np.max(gaussian)
            # perform the oversampling smoothness step
            image[no_support] = np.real(
                ifft2(
                    ifftshift(fourier_guess * gaussian)
                )
            )[no_support]

        # make current reconstruction to previous
        image_old = image_new
        # fourier transform the updated image
        fourier_guess = fftshift(fft2(image))
        # force the fourier amplitudes to conform the provided fourier plane
        fourier_update = fourier_amp * np.exp(1j * np.angle(fourier_guess))

    return image
