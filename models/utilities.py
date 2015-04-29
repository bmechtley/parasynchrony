"""
models/utilities.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Some helper utilities for multivariate spectra.
"""

import itertools

import numpy as np
import sympy

import scipy.signal
import matplotlib.mlab as mlab
import collections


class CacheDict(collections.OrderedDict):
    """
    Size-limited dictionary for caching results. See:
    http://stackoverflow.com/questions/2437617/
        limiting-the-size-of-a-python-dictionary
    """

    def __init__(self, *args, **kwargs):
        self.size_limit = kwargs.pop('size_limit', None)
        collections.OrderedDict.__init__(self, *args, **kwargs)
        self._check_size_limit()

    def __setitem__(self, key, value, dict_setitem=dict.__setitem__):
        collections.OrderedDict.__setitem__(
            self, key, value, dict_setitem=dict_setitem
        )
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


def correlation(x):
    """
    Return a normalized correlation matrix from a covariance matrix.

    :param x (np.array): input covariance matrix.
    :return (np.array): normalized correlation matrix with [1...] diagonal.
    """

    if np.any(np.isnan(x)):
        return np.full_like(x, np.nan)
    else:
        x = np.matrix(x)
        d = np.matrix(np.diag(x))
        return x / np.array(d.T * d) ** (1. / 2)

def solve_axatc(a, c):
    """
    Solves for X = A X A' + C
    See: http://math.stackexchange.com/questions/348487/solving-matrix-
        equations-of-the-form-x-axat-c

    :param a (np.array): A matrix.
    :param c (np.array): C matrix.
    :return: (np.array) X matrix.
    """

    a, c = [np.matrix(m) for m in a, c]

    evals, evecs = [np.matrix(m) for m in np.linalg.eig(a)]
    ones = np.matrix(np.ones(evecs.shape))

    phi = ones - evals.T * evals.getH().T
    gamma = np.linalg.inv(evecs) * c * np.linalg.inv(evecs).getH()
    x_tilde = np.multiply(1. / phi, gamma)

    return evecs * x_tilde * evecs.getH()


def smooth_phasors(x, magargs=None, phasorargs=None):
    """
    Smooth complex valued input by using separate smoothing kernels for
    magnitudes and phases. Smoothing on phases is done as smoothing on unit-
    length phasors.

    :param x (np.array): input array
    :param magargs (dict): arguments for smooth() on input magnitudes.
    :param phasorargs (dict): arguments for smooth() on input unit phasors.
    :return (np.array): smoothed data
    """

    return np.vectorize(complex)(
        smooth(abs(x), **magargs),
        np.angle(smooth(x / abs(x), **phasorargs))
    )


def smooth(x, window='boxcar', p=0.5, q=0.5):
    """
    Smooth input with a window that grows with the length of the array such
    that, as the number of dimensions approaches infinity, the number of values
    within the smoothing window approaches infinity, but the percentage of
    values in the window approaches zero.

    Window size is determined by: q * len(x)**p
    Note that p and q should be less than 1 for the above criteria to hold.

    :param x (np.array): timeseries data to smooth.
    :param window (str): window to grab from scipy.signal.get_window()
    :param p (float): linear scaling of window with length of x.
    :param q (float): polynomial scaling of window with length of x.
    :return (np.array): smoothed data
    """

    m = int(max(np.ceil((q * len(x))**p), 1))   # Window scale
    L = m * 2 + 1                               # Actual window size (odd)

    if window != 'median':
        win = scipy.signal.get_window(window, L)    # Window.

        # For edges, mirror the required amount of data around the edges such
        # that the smoothing window can begin with its center aligned with the
        # first data point and end with its center aligned with the last data
        # point.
        s = np.r_[x[m:0:-1], x, x[-2:-m-2:-1]]

        return scipy.signal.convolve(s, win / sum(win), mode='valid')
    else:
        return scipy.signal.medfilt(x, (L,))


def spectrum(series, **csdargs):
    """
    Compute spectral matrix for a multivariate time series. Uses Welch's method
    (scipy.signal.csd) of averaging over multiple FFT convolutions of sliding
    windows.

    :param series: (np.array) the multivariate (N,T) time series, with T
        timesteps.
    :param csdargs: (dict) additional parameters for matplotlib.mlab.csd.
    :return: (freqs, sxy), where freqs is an np.array of frequency values
        (0 to 0.5), and sxy is the (N,N,NFFT/2+1) np.array of cross-spectra.
    """

    nstates, nsamples = np.shape(series)

    # Default mlab.csd parameters.
    csdargs.setdefault('NFFT', 1024 if nsamples >= 1024 else nsamples)
    csdargs['NFFT'] = int(csdargs['NFFT'])
    csdargs.setdefault('Fs', 1)
    csdargs.setdefault('window', mlab.window_none)

    nbins = csdargs['NFFT'] / 2 + 1
    sxy = np.zeros((nstates, nstates, nbins), dtype=complex)
    freqs = np.linspace(0, csdargs.get('Fs') / 2.0, nbins)

    # Only compute cross-spectra once for combination of state variables.
    for i, j in itertools.combinations_with_replacement(range(nstates), 2):
        spec, _ = mlab.csd(series[i], series[j], **csdargs)

        # mlab.csd scales all values by a factor of two for total power in
        # one-sided FFT.
        sxy[i, j] = sxy[i, j] = spec[:, 0] / 2

    return freqs, sxy


def eval_matrix(m):
    """
    Evaluate a SymPy matrix and turn it into a numpy array. For whatever
    reason, this involves more than one would think, because each cell has to
    be converted back into a regular float.

    :param m: (sympy.Matrix) matrix to evaluate.
    :return: (np.array) resulting numpy array.
    """

    return np.array([
        complex(a) if sympy.im(a) != 0 else float(a) for a in m
    ]).reshape(m.shape)
