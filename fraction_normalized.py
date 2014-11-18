"""
fraction_normalized.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

This script is similar to fraction.py, but only shows fraction of synchrony for
host-to-host cospectra. It also shows a normalized cospectrum (normalized by
covariance) and a synchrony fraction based off that normalized cospectrum.
"""

# TODO: @ As approach Hopf bifurcation? PREPARE

import itertools

import numpy as np
import scipy.integrate

import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=6)
matplotlib.rc('lines', linewidth=0.5)
matplotlib.rc('axes', linewidth=0.5)
matplotlib.rc('font', family='serif')
import matplotlib.pyplot as pp

import models


def plot_shade_percentile(x, y, ax=None, perc=95, **kwargs):
    """
    Make a plot with values within the P-th percentile shaded in.

    :param x (np.array): x axis values
    :param y (np.array): y axis values
    :param ax (matplotlib.axes): axes on which to plot.
    :param perc (float): percentile to use (0-100).
    :param kwargs (dict): keyword arguments for ax.fill_between.
    """

    if ax is None:
        ax = pp.gca()

    perc = np.percentile(y, perc)

    ax.plot(x, y, **kwargs)
    ax.fill_between(
        x,
        np.ones(y.shape) * np.finfo(y.dtype).eps,
        y,
        where=y > perc,
        alpha=.25,
        **kwargs
    )


def plot_fraction(
        model, params, noisecov,
        cell=None, nfreqs=1000, plotarglist=None, ax=None
):
    """
    Make four plots for a set of model parameters. The first model is assumed
    to be the "baseline" model.

    1. Raw power spectrum for each model.
    2. Fraction of power spectra over the baseline model.
    3. Normalized spectrum for each model.
    4. Fraction of normalized spectrum over the baseline model.

    Plots will be placed within the axes specified by ax.

    :param model (models.StochasticModel): stochastic model specification.
    :param params (list): list of dictionaries containing parameter/value pairs.
    :param noisecov (list): list of numpy arrays for the noise covariance for
        each different set of parameters. Should be the same length as params.
    :param cell (tuple): combination of state variables for which to plot the
        cospectrum.
    :param nfreqs (int): number of points to sample the cospectrum at from 0
        to 0.5.
    :param plotarglist (list): list of dictionaries containing extra keywords
        to pass to plotting methods for each different set of parameters. Should
        be the same length as params.
    :param ax (list): list of four matplotlib.Axes instances in which to put the
        spectrum plots.
    """

    # Parameters as SymPy symbols.
    sym_params = [
        {
            models.Parasitism.params[name]: value
            for name, value in p.iteritems()
        } for p in params
    ]

    # The model's spectral matrix.
    freqs = np.linspace(0, 0.5, nfreqs)

    spectra = np.array([
        np.array([
            model.calculate_spectrum(sym_params[i], noisecov[i], v)
            for v in freqs
        ]).T
        for i in range(len(params))
    ])

    # Compute the normalization factors for the normalized cospectra by
    # integrating out the PSD numerically. Normalized cospectra should
    # integrate to the correlation coefficient, R^2.

    # TODO: @ This is a hack, integrating out the power density numerically
    # TODO:     instead of using an analytic solution.

    # The model's covariance (autocovariance at lag zero).
    # covariance = np.array([
    #     model.calculate_covariance(sym_params[i], noisecov[i])
    #     for i in range(len(params))
    # ])

    norms = [
        np.sqrt(np.prod([
            scipy.integrate.quad(
                lambda x: abs(
                    model.calculate_spectrum(
                        sym_params[i], noisecov[i], x
                    )
                )[a, a]**2, 0, 0.5
            )[0] * 2
            for a in cell
        ]))
        for i in range(len(params))
    ]

    # Different spectra for each set of model parameters. "raw" is the raw
    # spectral matrix, "pow" is the power spectrum, and "norm" is the
    # normalized cospectrum, Re(spectrum_ij) / sqrt(var(x_i)^2 * var(x_j)^2)

    spectra = [
        {
            'raw': spec[cell],
            'pow': abs(spec[cell])**2,
            'norm': np.real(spec[cell]) / norm
        }
        for i, (spec, norm) in enumerate(itertools.izip(spectra, norms))
    ]

    # Make the plots for each set of model parameters.
    for i, (spec, plotargs) in enumerate(itertools.izip(spectra, plotarglist)):
        # Subplot 1: Plot cospectrum.
        plot_shade_percentile(freqs, spec['pow'], ax=ax[0], **plotargs)

        # Subplot 3: Plot normalized cospectrum.
        plot_shade_percentile(freqs, spec['norm'], ax=ax[2], **plotargs)

        # Draw two lines: one for fraction of maximum, one for fraction of mean.
        if i != 0:
            identity = lambda x: x
            sync_fraction = lambda key, fun=identity: \
                fun(spec[key]) / fun(spectra[0][key]) * 100

            # Subplot 2. Plot fraction of synchrony from cospectrum.
            ax[1].plot(freqs, sync_fraction('pow'), **plotargs)
            ax[1].axhline(sync_fraction('pow', np.amax),  **plotargs)
            ax[1].axhline(sync_fraction('pow', np.mean), ls='--', **plotargs)

            # Subplot 4: Plot fraction of normalized cospectrum.
            ax[3].plot(freqs, sync_fraction('norm'), **plotargs)
            ax[3].axhline(sync_fraction('norm', np.amax), **plotargs)
            ax[3].axhline(sync_fraction( 'norm', np.mean), ls='--', **plotargs)

    # X-axis limit for all plots.
    for axi in range(4):
        ax[axi].set_xlabel('$\\omega$')
        ax[axi].set_xlim(freqs[0], freqs[-1])

    # Labels.
    varnames = tuple([str(model.vars[c]) for c in cell])

    ax[0].set_ylabel('$|f_{%s%s}|$' % varnames)
    ax[1].set_ylabel('$\\%% |f_{%s%s}|$' % varnames)
    ax[2].set_ylabel('$\\eta_{%s%s}$' % varnames)
    ax[3].set_ylabel('$\\%% \\eta_{%s%s}$' % varnames)


def main():
    # Initialize model / parameters.
    model = models.Parasitism.get_model("nbd(2)")

    baseparams = dict(
        r=3,
        a=0.5,
        c=1.2,
        k=0.9,
        mh=0.25,
        mp=0.25
    )

    evar = 0.5

    morans = [0, 0.25]
    migrations = [0, 0.25]
    cell = (0, 1)
    nfreqs = 2000

    basecov = np.identity(4) * evar
    basecov[0, 1] = basecov[1, 0] = basecov[2, 3] = basecov[3, 2] = morans[1]

    # Make figure.
    fig, ax = pp.subplots(16, 4, figsize=(8.5, 22))

    # Plot each combination of model parameters against the "baseline" model,
    # which uses all sources of synchrony, i.e. moran/dispersal on both host
    # and parasitoid.

    for ri, (migh, migp, morh, morp) in enumerate(itertools.product(
        morans, morans, migrations, migrations
    )):
        print ri, migh, migp, morh, morp

        cov = np.array([
            [evar, morh, 0, 0],
            [morh, evar, 0, 0],
            [0, 0, evar, morp],
            [0, 0, morp, evar]
        ])

        params = dict(baseparams)
        params['mh'] = migh
        params['mp'] = migp

        ax[ri][0].set_title('Moran %s %s Dispersal %s %s' % (
            'H' if morh else '-',
            'P' if morp else '-',
            'H' if migh else '-',
            'P' if migp else '-'
        ))

        plot_fraction(
            model,
            [baseparams, params],
            [basecov, cov],
            cell=cell,
            nfreqs=nfreqs,
            plotarglist=[{'color': 'k'}, {'color': 'r'}],
            ax=ax[ri]
        )

    pp.subplots_adjust(wspace=0.4, hspace=0.8, top=0.95, bottom=0.05)
    pp.savefig('fraction_normalized.pdf')

if __name__ == '__main__':
    main()
