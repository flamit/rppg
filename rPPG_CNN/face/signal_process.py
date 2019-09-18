import numpy as np
from scipy.sparse import spdiags


def detrend(signal, Lambda):
    """applies a detrending filter.

    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.

    Parameters
    ----------
    signal: numpy.ndarray
    The signal where you want to remove the trend.
    Lambda: int
    The smoothing parameter.

    Returns
    -------
    filtered_signal: numpy.ndarray
    The detrended signal.

    """
    signal_length = signal.shape[0]

    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2*np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data,
                diags_index, (signal_length-2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda**2) * np.dot(D.T, D))), signal)
    return filtered_signal
