"""
models.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Module for various stochastic difference equation models. In addition to a few
helper functions (mostly involved with plotting simulated results), this module
defines a StochasticModel class which is a generalized class for stochastic
difference equations, allowing the automatic analytic linearization of
sufficiently simple nonlinear models using SymPy. Additionally, the class
Parasitism contains several class methods for constructing prefab
StochasticModels, e.g. stochastic AR-1 processes, Nicholson-Bailey, and
Nicholson-Bailey with negative binomial functional response, as well as a
constructor that can use any of these single-patch models to create a
multipatch model with global dispersal.
"""

import string
import sympy as sym
import numpy as np
import scipy as sp
import itertools
import scipy.signal

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage{xfrac}')
import matplotlib.colors as mplcolors
import matplotlib.mlab as mlab
import matplotlib.pyplot as pp
import matplotlib.gridspec as gridspec


def symlog(x):
    """
    Signed log-scaling of input by its distance from zero. log(abs(x)) * sign(x)

    :param x (float): input value.
    :return: symmetrically log-scaled value.
    """

    dtype = type(x) if type(x) != np.ndarray else x.dtype

    return np.log(abs(x) + np.finfo(dtype).eps) * np.sign(x)


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

        # For edges, mirror the required amount of data around the edges such that
        # the smoothing window can begin with its center aligned with the first
        # data point and end with its center aligned with the last datapoint.
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
    sxy = np.empty((nstates, nstates, nbins), dtype=complex)
    freqs = np.linspace(0, csdargs.get('Fs') / 2.0, nbins)

    # Only compute cross-spectra once for combination of state variables.
    for i, j in itertools.combinations_with_replacement(range(nstates), 2):
        spec, _ = mlab.csd(series[i], series[j], **csdargs)

        # TODO: @ mlab.csd scales all values by a factor of two. Not sure why.
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

    return np.array([float(a) for a in m.evalf()]).reshape(m.shape)


def plot_twopane_axes(
        n=1,
        ylabels=None,
        yticks=None,
        ylimits=None
):
    """
    Set up a GridSpec for an NxN lower left triangular grid of subplots, each
    of which has two vertically stacked panes.

    :param n: Number of rows/columns in the grid.
    :param ylabels (nested list): NxN list of dictionaries, where each dict has
        a 'top' and 'bottom' key, the values of which are the ylabels for the
        corresponding panes (default: None).
    :param yticks (dict): dictionary with 'top' and 'bottom' keys indicating y
        tick values for top and bottom panes, assuming they are the same across
        all grid cells (default: None).
    :param ylimits (dict): dictionary with 'top' and 'bottom' keys pointing to
        two-value tuples indicating lower and upper y axis limits, assuming
        they are the same across all grid cells (default: None).
    :return (list): list of axes of instance matplotlib.Axes.
    """

    fig = pp.gcf()
    gs = gridspec.GridSpec(n, n, wspace=0.25, hspace=0.25)
    axlist = [[{} for j in range(n)] for i in range(n)]

    for i, j in itertools.combinations_with_replacement(range(n), 2):
        # This is the vertically stacked grid inside the current grid cell.
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[n * j + i],
            wspace=0, hspace=0
        )

        for pane_index, pane in enumerate(['top', 'bottom']):
            ax = pp.Subplot(pp.gcf(), inner_grid[pane_index])

            # Both panes should share the lower pane's x axis.
            if pane is 'top':
                ax.get_xaxis().set_visible(False)

            fig.add_subplot(ax)
            pp.sca(ax)

            # Set y ticks, limits, and labels if they're specified.
            if yticks[pane] is not None:
                pp.yticks(yticks[pane].values())
                ax.set_yticklabels(yticks[pane].keys())

            if ylimits is not None and pane in ylimits:
                pp.ylim(*ylimits[pane])

            if ylabels is not None:
                pp.ylabel(ylabels[i][j][pane])

            axlist[i][j][pane] = ax

    return axlist


def plot_cospectra(
        freqs,
        pxx,
        varnames=None,
        plotfun='auto',
        axpanes=None,
        **plotargs
):
    """
    Make a NxN lower triangular plot of cospectra between state variables.

    :param freqs (array): (F,) List of frequencies.
    :param pxx (array): The (N,N,F) spectral matrix.
    :param varnames (list): List of N names for the states. If None, no labels
        will be drawn (default: None).
    :param plotfun (callable or str): Plot function to use for the spectrum. If
        'auto', pyplot.scatter will be used for frequency resolution F <= 8192,
        and pp.hist2d will be used for F > 8192 (default: 'auto').
    :param axpanes (list): nested NxN list of dictionaries where each dict has
        keys 'top' and 'bottom' that point to matplotlib.Axes instances
        corresponding to the top (magnitude) and bottom (phase) plots
        respectively. If None is specified, the axes will be created and
        returned for re-use in overlaying plots.
    :param plotargs (dict): additional arguments to pass to plotfun.
    :return (list): nested NxN list of dictionaries formed in the same way as
        axpanes.
    """

    # Set up defaults for plotting type/parameters.
    if plotfun is None:
        plotfun = pp.plot
    elif plotfun is 'auto':
        # If auto, automatically determine if a plot should use scatter or
        # hist2d based on how many bins there are to plot.
        if len(freqs) > 8192:
            plotfun = pp.hist2d
        else:
            plotfun = pp.scatter

    if plotfun is pp.hist2d:
        plotargs.setdefault('bins', 200)
        plotargs.setdefault('normed', True)

        # If "color" is specified, make a colormap that linearly scales alpha
        # at that color. Otherwise, use the default or specified cmap. Useful
        # for overlaying multiple plots with different colors.
        if 'color' in plotargs:
            if 'cmap' not in plotargs:
                plotargs['cmap'] = mplcolors.LinearSegmentedColormap.from_list(
                    'custom_histcmap',
                    [
                        mplcolors.colorConverter.to_rgba(plotargs['color'], 0),
                        mplcolors.colorConverter.to_rgba(plotargs['color'], 1)
                    ]
                )

            plotargs.pop('color')
    elif plotfun is pp.scatter:
        plotargs.setdefault('alpha', 0.25)
        plotargs.setdefault('marker', '.')
        plotargs.setdefault('s', 1)

    # Make axes for two-pane lower left triangle grid if they aren't provided.
    n = len(pxx)

    if axpanes is None:
        axpanes = plot_twopane_axes(
            n=n,
            ylabels=[
                [
                    dict(
                        top='$\\log f_{%s%s}$' % (varnames[i], varnames[j]),
                        bottom='$\\angle f_{%s%s}$' % (varnames[i], varnames[j])
                    )
                    for j in range(n)
                ]
                for i in range(n)
            ] if varnames is not None else None,
            yticks=dict(
                top=None,
                bottom=dict(zip(
                    ['$-pi/2$', '$0$', '$pi/2$'],
                    [-np.pi/2, 0, np.pi/2]
                ))
            ),
            ylimits=dict(bottom=[-np.pi, np.pi])
        )

    # Plot each spectral matrix component in the created/provided axes.
    for i, j in itertools.combinations_with_replacement(range(n), 2):
        # Magnitude.
        pp.sca(axpanes[i][j]['top'])
        plotfun(freqs, abs(pxx[i, j]), **plotargs)

        if np.sum(abs(pxx[i, j])) > 0:
            pp.yscale('log')

        pp.xlim(0, freqs[-1])

        # Phase.
        pp.sca(axpanes[i][j]['bottom'])
        plotfun(freqs, np.angle(pxx[i, j]), **plotargs)
        pp.xlim(0, freqs[-1])

    return axpanes


def plot_phase(
        series,
        varnames=None,
        logscale=False,
        plotfun='auto',
        **plotargs
):
    """
    Make a NxN lower triangular phase-space plot that plots pairs of state
    variables as functions of each other in a two-dimensional histogram.

    :param series (array): The (N,T) state trajectory.
    :param varnames (list): N state variable names. If None, no labels will be
        drawn (default: None).
    :param logscale (bool): Whether or not variables should be plot on a log
        scale (default: False).
    :param plotargs: Any additional parameters to pyplot.hist2d, such as bin
        count.
    """

    # Set up defaults for plotting type/parameters.
    if plotfun is None:
        plotfun = pp.scatter
    elif plotfun is 'auto':
        if len(series.shape) > 1 and series.shape[1] > 8192:
            plotfun = pp.hist2d
        else:
            plotfun = pp.scatter

    if plotfun is pp.scatter:
        plotargs.setdefault('alpha', 0.25)
        plotargs.setdefault('marker', '.')
        plotargs.setdefault('s', 1)
    elif plotfun is pp.hist2d:
        plotargs.setdefault('bins', 200)
        plotargs.setdefault('normed', True)

        if 'cmap' not in plotargs:
            plotargs.setdefault('color', 'green')

            plotargs['cmap'] = mplcolors.LinearSegmentedColormap.from_list(
                'custom_histcmap',
                [
                    mplcolors.colorConverter.to_rgba(plotargs['color'], 0),
                    mplcolors.colorConverter.to_rgba(plotargs['color'], 1)
                ]
            )
            plotargs.pop('color')

    nstates, nsamples = series.shape

    # Plot lower left triangle grid.
    for i, j in itertools.combinations_with_replacement(range(nstates), 2):
        pp.subplot(nstates, nstates, nstates * j + i + 1)

        # Plot axis labels if varname are provided.
        if varnames is not None:
            if j is len(varnames) - 1:
                pp.xlabel('$%s$' % varnames[i])

            if i is 0:
                pp.ylabel('$%s$' % varnames[j])

        # Scale values.g
        xs, ys = series[i], series[j]
        if logscale: xs, ys = symlog(xs), symlog(ys)

        plotfun(xs, ys, **plotargs)


class StochasticModel:
    """
    Describes a system of stochastic difference equations with a set of
    state variables, noise variables sampled from a multinormal distribution,
    and a matrix expression as a function of both. Variables are SymPy symbols,
    and the dynamics are expressed as a SymPy Matrix object.
    """

    # TODO: @ Try using stattools to create a vector AR process instead of
    # TODO:     manually computing properties of the linear model. I'm not sure
    # TODO:     how complete stattools's ARMA implementation is for the
    # TODO:     multivariate case, though.

    # TODO: @ look at greenman and benton for magnitude of peak

    def __init__(self, symvars, noises, equation):
        """
        Initialize a model object.
        :param symvars: list of SymPy symbols representing state variables.
        :param noises: list of SymPy symbols representing noise variables.
        :param equation: SymPy Matrix describing dynamics of the state
            variables. X_t+1 = F(X_t)
        """

        self.vars = sym.Matrix(symvars)
        self.noises = sym.Matrix(noises)
        self.stochastic = sym.Matrix(equation)
        self.deterministic = self.stochastic.subs(
            dict(zip(noises, [0] * len(noises)))
        )

        self.equilibrium = None
        self.m1 = None
        self.q0 = None

        self.simulated = None
        self.simulated_linear = None

        # Cache for matrices evaluated for certain parameters. The cache is
        # indexed by a hash of the dictionary of parameters.
        self.cache = {}

    def solve_equilibrium(self):
        """Find the [first] non-trivial equilibrium point."""

        # TODO: @ This currently assumes there's one non-trivial equilibrium
        # TODO:     point and that it's the second returned by SymPy's solve().
        # TODO:     That's a dangerous assumption.

        eq = sym.solve(
            sym.Eq(self.vars, self.deterministic),
            self.vars,
            dict=True
        )

        self.equilibrium = eq[-1]

        return eq[-1]

    def linearize(self):
        """
        Create deterministic / stochastic transform matrices for linearization
        about the equilibrium point. Note that the equilibrium must be set
        before this is called.  This creates two matrices, m1 and q0. m1 is the
        deterministic transition matrix for the linear model and q0 is the
        noise-dependence matrix.
        """

        # Use equilibrium with 0 noise terms for substitutions.
        subs = dict(self.equilibrium.items())
        subs.update({noise: 0 for noise in self.noises})

        self.m1 = self.deterministic.jacobian(self.vars).subs(subs)
        self.q0 = self.stochastic.jacobian(self.noises).subs(subs)

    def get_cached_matrices(self, params):
        """
        Cache m1/q0 matrices for the given parameter values to avoid
        recomputing each time we want to compute spectral properties for the
        same parameter values (e.g. computing the full spectral matrix at
        multiple frequencies).

        :param params (dict): dictionary of parameter values with SymPy symbol
            keys.
        :return (dict): dictionary with two keys, 'm1' and 'q0', each
            containing a numpy array for the matrix evaluated at the parameter
            values.
        """

        if self.m1 is None or self.q0 is None:
            self.linearize()

        phash = hash(frozenset(params.items()))
        self.cache.setdefault(phash, {})
        cached = self.cache[phash]

        if 'm1' not in cached:
            cached['m1'] = eval_matrix(self.m1.subs(params))

        if 'q0' not in cached:
            cached['q0'] = eval_matrix(self.q0.subs(params))

        return cached

    def calculate_covariance(self, params, covariance):
        """
        Calculate the covariance (autocovariance with lag zero) for the model.

        :param params (dict): parameter values with SymPy symbol keys.
        :param covariance (np.array): covariance of the noise.
        :return: (np.array): covariance matrix.
        """

        # TODO: Covariance: verify this result.

        cached = self.get_cached_matrices(params)
        q0, m1 = cached['q0'], cached['m1']

        return q0 * covariance * q0.T * np.linalg.inv(
            2 * np.identity(len(m1)) - 0.5 * m1.T * m1
        )

    def calculate_eigenvalues(self, params):
        """
        Calculate the dominant frequency of oscillation for the linearized
        model. This will be the same for all cross-spectra.

        :param params (dict): dictionary of parameter values with SymPy symbol
            keys.
        :return (float): frequency of the system's oscillation.
        """

        cached = self.get_cached_matrices(params)
        return np.linalg.eig(cached['m1'])

    def calculate_spectrum(self, params, covariance, v=0):
        """
        Calculate the spectral matrix according to the model linearized around
        its equilibrium. Note that linearize() must be called before this.

        :param params: (dict) free parameters to the model (excluding
            state/noise). Keys are SymPy symbols.
        :param covariance: (np.array) covariance of the noise, assuming each
            noise parameter is a dimension of a multivariate normal, with
            dimensions ordered according to self.noises.
        :return: NxN matrix of co-spectra, where N is the number of state
            variables.
        """

        cached = self.get_cached_matrices(params)

        # Mu is the spectral component.
        mu = np.exp(-2j * np.pi * v)

        if cached['m1'].shape[0] < 2:
            r1 = np.array([(1 - mu * cached['m1'][0, 0])**-1])
            r1.reshape((1, 1))
        else:
            r1 = sp.linalg.inv(
                np.identity(len(cached['m1'])) - mu * cached['m1']
            )

        r = np.dot(r1, cached['q0'])

        return np.dot(np.dot(r,  covariance), np.conj(r.T))

    def simulate(self, initial, params, covariance, timesteps=1000):
        """
        Start from an initial point and simulate the model with sampled noise.

        :param initial: (list) Initial values for the state variables, ordered
            according to self.vars.
        :param params: (dict) free parameters to the model (excluding state/
            noise). Keys are SymPy symbols.
        :param covariance: (np.array) covariance of the noise, assuming each
            noise parameter is a dimension of a multivariate normal, with
            dimensions ordered according to self.noises.
        :param timesteps: (int) number of timesteps to simulate.
        :return: NxT matrix of state vectors, where N is the number of
            variables and T is the number of timesteps.
        """

        nstates = len(initial)

        params = dict(params.items())

        # Begin the model at the equilibrium.
        params.update({noise_var: 0 for noise_var in self.noises})
        params.update({
            state_var: self.equilibrium[state_var]
            for state_var in self.vars
        })

        param_keys = params.keys()

        # Use sympy.utilities.lambdify to set up a lambda function to quickly
        # evaluate the model at each timestep. Lamdify just evaluates a string
        # of Python code, so first translate all variables into Pythno-friendly
        # variable names by replacing any special LaTeX characters with
        # underscores.
        trantab = string.maketrans(r'{}\()^*[]', r'_________')
        fixed_param_keys = [str(k).translate(trantab) for k in param_keys]

        subs = dict(zip(param_keys, fixed_param_keys))
        param_inds = {param_keys[i]: i for i in range(len(param_keys))}
        param_vals = [params[k] for k in param_keys]

        f = sym.utilities.lambdify(
            fixed_param_keys,
            self.stochastic.subs(subs)
        )

        # Simulate the model.
        rnoise_mean = np.zeros((len(covariance),))
        self.simulated = np.zeros((timesteps, len(initial)))
        self.simulated[0] = initial

        for step in range(1, timesteps):
            # Generate random noise with the given covariance.
            rnoise = np.random.multivariate_normal(rnoise_mean, covariance)

            # Fill the model parameters with the output of the last iteration.
            for i, noise_var in enumerate(self.noises):
                param_vals[param_inds[noise_var]] = rnoise[i]

            for i, state_var in enumerate(self.vars):
                param_vals[param_inds[state_var]] = self.simulated[step - 1, i]

            # Evaluate the model at this timestep.
            sim = f(*param_vals)

            self.simulated[step] = np.reshape(sim, (nstates,))

        return self.simulated.T

    def simulate_linear(self, initial, params, covariance, timesteps=1000):
        """
        Start from an initial point and simulate the model linearized about
        the equilibrium with sampled noise.

        :param initial: (list) Initial values for the state variablees, ordered
            according to self.vars.
        :param params: (dict) free parameters to the model (excluding state/
            noise). Keys are SymPy symbols.
        :param covariance: (np.array) Covariance of the noise, assuming each
            noise parameter is a dimension of a multivariate normal, with
            dimensions ordered according to self.noises.
        :param timesteps: (int) number of timesteps to simulate.
        :return: NxT matrix of state vectors, where N is the number of
            variables and T is the number of timesteps.
        """

        # Evaluate the M1 and Q0 matrices with zero noise variance at the
        # equilibrium.

        params = dict(params.items())
        params.update({noise_var: 0 for noise_var in self.noises})
        params.update({
            state_var: self.equilibrium[state_var]
            for state_var in self.vars
        })

        m1f = eval_matrix(self.m1.subs(params))
        q0f = eval_matrix(self.q0.subs(params))

        # Simulate the model.
        self.simulated_linear = np.zeros((timesteps, len(initial)))
        self.simulated_linear[0] = initial

        rnoise_mean = np.zeros((len(covariance),))

        for step in range(1, timesteps):
            # Generate random noise with the given covariance.
            if np.sum(np.abs(covariance)) == 0:
                rnoise = np.zeros((len(covariance),))
            else:
                rnoise = np.random.multivariate_normal(rnoise_mean, covariance)

            linear_term = np.dot(m1f, self.simulated_linear[step - 1])
            stochastic_term = np.dot(q0f, rnoise)

            self.simulated_linear[step] = linear_term + stochastic_term

        return self.simulated_linear.T


class Parasitism:
    """
    Class that contains several StochasticModel constructors that create models
    representing host-parasitoid interactions.
    """

    # TODO: @ organize h-p-h-p so that my jacobian is a block-symmetric matrix
    # TODO:     where each block is a patch

    # TODO: @ look up eigenvalues of block matrices (det(A)det(B)-det(C)det(D))
    # TODO:     [AB;CD]

    # TODO: @ extend main theorem to come up with lemmas for spectral peaks /
    # TODO:     magnitudes, possibly for special cases. Simplify these in terms
    # TODO:     of general parameters for any model that includes symmetric
    # TODO:     migration and specifically for each model, to factor out any
    # TODO:     parameters that do not affect either the location of the peak
    # TODO:     or its magnitude.

    params = dict(
        a=sym.Symbol('a', positive=True),
        r=sym.Symbol('\lambda', positive=True),
        c=sym.Symbol('c', positive=True),
        k=sym.Symbol('k', positive=True),
        mh=sym.Symbol('\mu_H', positive=True),
        mp=sym.Symbol('\mu_P', positive=True),
        h=[sym.Symbol('H')] + list(sym.symbols('H^{((1:3))}')),
        p=[sym.Symbol('P')] + list(sym.symbols('P^{((1:3))}')),
        eh=[sym.Symbol('\epsilon_h')] + list(
            sym.symbols('\epsilon^{((1:3))}_h')
        ),
        ep=[sym.Symbol('\epsilon_p')] + list(
            sym.symbols('\epsilon^{((1:3))}_p')
        ),
        alpha=sym.Symbol(r'\alpha'),
        x=sym.Symbol('x'),
        e=sym.Symbol(r'\epsilon')
    )

    @staticmethod
    def ar1():
        """
        AR-1 autoregressive model for one state variable,
            x_t = \alpha x_{t-1} + e
        :return: StochasticModel for AR1 process.
        """

        a, x, e = [Parasitism.params[v] for v in ['alpha', 'x', 'e']]

        return StochasticModel([x], [e], [a * x + e])

    @staticmethod
    def nb():
        """
        Single-patch Nicholson-Bailey model with no regularization. Note: this
        has an unstable equilibrium.
            H_t = \lambda H_{t-1} exp(-a P_{t-1}) exp(e_H)
            P_t = c H_{t-1} (1 - exp(-a P_{t-1})) exp(e_P)
        :return: StochasticModel for Nicholson-Bailey process.
        """
        h, p, eh, ep, r, a, c = [Parasitism.params[v] for v in [
            'h', 'p', 'eh', 'ep', 'r', 'a', 'c'
        ]]

        return StochasticModel(
            [h[0], p[0]],
            [eh[0], ep[0]],
            [
                r * h[0] * sym.exp(-a * p[0]) * sym.exp(eh[0]),
                c * h[0] * (1 - sym.exp(-a * p[0])) * sym.exp(ep[0])
            ]
        )

    @staticmethod
    def nbd():
        """
        Single-patch Nicholson-Bailey model with negative binomial functional
        response.
            H_t = \lambda H_{t-1} (1 + a \frac{P_{t-1}}{K})^{-k} exp(e_H)
            P_t = c H_{t-1} (1 - (1 + a \frac{P_{t-1}}{K})^{-k}) exp(e_P)
        :return: StochasticModel for Nicholson-Bailey process with negative
            binomial functional response.
        """
        h, p, eh, ep, r, a, c, k = [Parasitism.params[v] for v in [
            'h', 'p', 'eh', 'ep', 'r', 'a', 'c', 'k'
        ]]

        return StochasticModel(
            [h[0], p[0]],
            [eh[0], ep[0]],
            [
                r * h[0] * (1 + a * p[0] / k) ** -k * sym.exp(eh[0]),
                c * h[0] * (1 - (1 + a * p[0] / k) ** -k) * sym.exp(ep[0])
            ]
        )

    @staticmethod
    def get_model(modelstr):
        """
        Get a model by name. For each constructor in Parasitism (e.g. ar1, nb,
        nbd), the name can be taken along with an integer parameter indicating
        the number of patches to use for a global dispersal matrix parameterized
        by m_H and m_P, where each is the probability that a member of the
        population will move to another patch. Dispersal is global and
        symmetric, so each patch obtains equal migration from the others.

        :param modelstr (str): String indicating the model and number of
            patches as "model(N)", e.g. "nb(1)" or "nbd(2)". If no parameter
            is specified (i.e. just "model"), a single-patch model will be
            returned.
        :return: StochasticModel for the specified multipatch model.
        """

        h, p, eh, ep, r, a, c, k, mp, mh = [Parasitism.params[v] for v in [
            'h', 'p', 'eh', 'ep', 'r', 'a', 'c', 'k', 'mp', 'mh'
        ]]

        # Tokenize model string to obtain the specified model.
        tokens = modelstr.split('(')
        mname = tokens[0]
        refmodel = getattr(Parasitism, mname)()
        refmodel.solve_equilibrium()
        refmodel.linearize()

        if len(tokens) < 2:
            # If number of patches is not specified, return the one-patch model.
            return refmodel
        else:
            # Make a multipatch model from the reference model with global
            # dispersal.

            params = tokens[1].rstrip(')').split(',')
            npatches = int(params[0])
            migration = params[1] if len(params) > 1 else 'global'

            # migrationmat will be the transition matrix applied to the
            # difference equations for population dynamics in each patch.
            # For global dispersal, it will have 1-m in the diagonal and m / N
            # in the off diagonals such that hosts only migrate to other hosts
            # and parasitoids only migrate to other parasitoids.
            #
            # TODO: @ This will NOT work, e.g. with AR1 or any non host-
            # TODO:     parasitoid model or multispecies model. Only for models
            # TODO:     where there are one host and one parasitoid per patch,
            # TODO:     and hosts precede parasitoids in the list of state
            # TODO:     variables.
            migrationmat = sym.zeros(npatches * 2)

            if migration == 'global':
                for row in range(npatches * 2):
                    for col in range(npatches * 2):
                        m = mh if row < npatches else mp

                        if row == col:
                            migrationmat[row, col] = 1-m
                        elif row < npatches and col < npatches:
                            # Host migration.
                            migrationmat[row, col] = m / npatches
                        elif row >= npatches and col >= npatches:
                            # Parasitoid migration.
                            migrationmat[row, col] = m / npatches

            # Ordinary difference equations to which migration matrix will be
            # applied.
            equations = sym.Matrix([
                item for sublist in [
                    [
                        refmodel.stochastic[i].subs({
                            h[0]: h[j+1],
                            p[0]: p[j+1],
                            eh[0]: eh[j+1],
                            ep[0]: ep[j+1]
                        }) for j in range(npatches)
                    ] for i in range(2)
                ] for item in sublist
            ])

            # Create the model.
            multipatch = StochasticModel(
                [
                    h[i+1] for i in range(npatches)
                ] + [
                    p[i+1] for i in range(npatches)
                ],
                [
                    eh[i+1] for i in range(npatches)
                ] + [
                    ep[i+1] for i in range(npatches)
                ],
                migrationmat * equations
            )

            # Solve for the equilibrium. Assuming global dispersal, the
            # equilibrium should be uniform across all patches and not depend
            # on the migration parameters.

            if refmodel.equilibrium is not None:
                multipatch.equilibrium = {
                    h[i+1]: refmodel.equilibrium[h[0]]
                    for i in range(npatches)
                }

                multipatch.equilibrium.update({
                    p[i+1]: refmodel.equilibrium[p[0]]
                    for i in range(npatches)
                })

            # Linearize the model.
            multipatch.linearize()

            return multipatch
