"""
fraction_normalized.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

This script is similar to fraction.py, but only shows fraction of synchrony for
host-to-host cospectra. It also shows a normalized cospectrum (normalized by
covariance) and a synchrony fraction based off that normalized cospectrum.
"""

# TODO: @ Normalized cospectrum, normalized by geometric mean of variance of
# TODO:     the two populations. Correlation integrated across freqs is R^2,
# TODO:     the correlation coefficient.
# TODO: @ 1. normalized covariance, 2. fraction of normalized covariance
# TODO:     normalized cospectrum in Dan's doc.--pointwise normalized is other.

# TODO: @ As approach Hopf bifurcation? PREPARE

import itertools

import numpy as np

import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=6)
matplotlib.rc('lines', linewidth=0.5)
matplotlib.rc('axes', linewidth=0.5)
matplotlib.rc('font', family='serif')
import matplotlib.pyplot as pp

import models


def plot_shade_percentile(x, y, ax=None, **kwargs):
    if ax == None: ax = pp.gca()

    perc = np.percentile(y, 95)

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

    # The model's covariance (autocovariance at lag zero).
    covariance = np.array([
        model.calculate_covariance(sym_params[i], noisecov[i])
        for i in range(len(params))
    ])

    r, c = cell

    # The spectrum for the full model (first in the list of params, noisecov)
    full_spectrum = spectra[0][cell]
    rawfull_mag = abs(full_spectrum)
    rawfull_re = np.real(full_spectrum)
    normfull_re = rawfull_re / np.sqrt(
        covariance[0, r, r] * covariance[0, c, c]
    )

    for i, (spec, mcov, plotargs) in enumerate(itertools.izip(
        spectra, covariance, plotarglist
    )):
        r, c = cell

        rawspec_mag = abs(spec[cell])
        rawspec_re = np.real(spec[cell])
        normspec_re = rawspec_re / np.sqrt(mcov[r, r] * mcov[c, c])

        # Subplot 1: Plot cospectrum.
        plot_shade_percentile(freqs, rawspec_mag, ax=ax[0], **plotargs)

        # Subplot 3: Plot normalized cospectrum.
        plot_shade_percentile(freqs, normspec_re, ax=ax[2], **plotargs)

        # Draw two lines: one for fraction of maximum, one for fraction of mean.
        if i != 0:
            # Subplot 2. Plot fraction of synchrony from cospectrum.
            ax[1].plot(freqs, rawspec_mag / rawfull_mag * 100, **plotargs)
            ax[1].axhline(
                np.amax(rawspec_mag) / np.amax(rawfull_mag) * 100, **plotargs
            )
            ax[1].axhline(
                np.mean(rawspec_mag) / np.mean(rawfull_mag) * 100,
                ls='--', **plotargs
            )

            # Subplot 4: Plot fraction of normalized cospectrum.
            ax[3].plot(freqs, normspec_re / normfull_re * 100, **plotargs)
            ax[3].axhline(
                np.amax(normspec_re) / np.amax(normfull_re) * 100, **plotargs
            )
            ax[3].axhline(
                np.mean(normspec_re) / np.mean(normfull_re) * 100,
                ls='--', **plotargs
            )

    # x axis limit for all plots.
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
    model = models.Parasitism.get_model("nbd(2)")

    baseparams = dict(
        r=3,
        a=0.5,
        c=1.2,
        k=0.9,
        mh=0.5,
        mp=0.5
    )

    evar = 0.5
    basecov = np.identity(4) * evar

    morans = [0, 0.25]
    migrations = [0, 0.25]
    cell = (0, 1)
    nfreqs = 1000

    fig, ax = pp.subplots(16, 4, figsize=(8.5, 22))

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
