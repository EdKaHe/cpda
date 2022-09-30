from scipy.interpolate import interp1d


def interp_alpha_1d(sample_size, sample_alpha, interp_size, axis=0, kind="cubic", fill_value="extrapolate"):
    # interpolate the polarizabilities
    interp_alpha = interp1d(sample_size, sample_alpha, axis=axis, kind=kind, fill_value=fill_value)

    # calculate the polarizabilities at the interpolation points
    interp_alpha = interp_alpha(interp_size)

    return interp_alpha.reshape(-1, sample_alpha.shape[1 - axis])