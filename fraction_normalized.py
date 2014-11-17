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


def plot_fraction(
        model, params, covariance,
        cell=None, nfreqs=1000, plotarglist=None, ax=None
):
    sym_params = [
        {
            models.Parasitism.params[name]: value
            for name, value in p.iteritems()
        } for p in params
    ]

    freqs = np.linspace(0, 0.5, nfreqs)

    spectra = np.array([
        np.array([
            model.calculate_spectrum(sym_params[i], covariance[i], v)
            for v in freqs
        ]).T
        for i in range(len(params))
    ])

    magnitude_spectra = abs(spectra)

    #
    # eigvals, eigvecs = model.calculate_eigenvalues(sym_params)
    # oscfreqs = np.angle(eigvals) / (2 * np.pi)
    # oscmags = abs(eigvals)
    #
    # Plot vertical line at the dominant frequency in both subplots.
    # for mag, freq in itertools.izip(oscmags[ctuple], oscfreqs[ctuple]):
    #     ax[i, sp].axvline(freq, color='green')
    #     ax[i, sp+1].axvline(freq, color='green')
    #

    # 1. Plot cospectrum.
    for magspec, plotargs in itertools.izip(magnitude_spectra, plotarglist):
        ax[0].plot(freqs, magspec[cell], **plotargs)
        perc = np.percentile(magspec[cell], 95)
        ax[0].fill_between(
            freqs,
            np.ones(magspec[cell].shape) * np.finfo(magspec[cell].dtype).eps,
            magspec[cell],
            where=magspec[cell] > perc,
            alpha=.25,
            **plotargs
        )

    # 2. Plot fraction of synchrony from cospectrum.
    for magspec, plotargs in itertools.izip(magnitude_spectra, plotarglist):
        ax[1].plot(
            freqs, magspec[cell] / magnitude_spectra[0][cell] * 100, **plotargs
        )

        ax[1].axhline(
            np.amax(magspec[cell]) / np.amax(magnitude_spectra[0][cell]) * 100,
            **plotargs
        )

        ax[1].axhline(
            np.mean(magspec[cell]) / np.mean(magnitude_spectra[0][cell]) * 100,
            ls='--', **plotargs
        )

    # x axis limit for all plots.
    for axi in range(4):
        ax[axi].set_xlabel('$\omega$')
        ax[axi].set_xlim(freqs[0], freqs[-1])

    # Labels.
    varnames = [str(model.vars[c]) for c in cell]

    ax[0].set_yscale('log')
    ylabel = '$|f_{%s%s}|$' % (varnames[0], varnames[1])
    ax[0].set_ylabel(ylabel)

    ax[1].set_ylabel('$\\%% |f_{%s%s}|$' % (varnames[0], varnames[1]))
    ax[1].set_ylim(0, 110)


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
            plotarglist=[
                dict(color='k'),
                dict(color='r')
            ],
            ax=ax[ri]
        )

    pp.subplots_adjust(wspace=0.4, hspace=0.8, top=0.95, bottom=0.05)
    pp.savefig('fraction_normalized.pdf')

if __name__ == '__main__':
    main()
