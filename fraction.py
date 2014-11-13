"""
fraction.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

This is a quick script to demonstrate the fraction of synchrony for which
each possible source is responsible in the two-patch negative binomial model.
In this model, there are four possible exogenous sources of synchrony,
including combinations of whether or not the Moran effect and migration are
present for the host and parasitoid pairs, respectively.

This makes for a total of 16 different possible models. For each model,
the host-to-host and parasitoid-to-parasitoid cross spectra are plotted in red,
and the cross spectrum from the full model (including all exogenous sources of
sychrony) is plotted in black. Next to each plot, the ratio of the model's
cross-spectra to the full model's cross-spectra is expressed as a percentage.

Note that at times this percentage goes above 100%, as some models introduce
more synchrony between these variables than does the full model.
"""

# TODO: @ Verify numerical result for matrix inverse. Get inverse, multiply by
# TODO:     original, see if I get same result to N decimal places. If I don't,
# TODO:     throw an error.

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

# 1. Model parameters.
# TODO: @ Maybe use pybatchdict for these.
model = models.Parasitism.get_model("nbd(2)")
params = np.array([
    [
        dict(
            r=3.0,
            a=0.5,
            c=1.2,
            k=0.9,
            mh=migration_host,
            mp=migration_parasitoid
        ) for migration_parasitoid in [0, 0.05]
    ] for migration_host in [0, 0.05]
])

covariance = np.array([
    [
        [
            [1E-2, moran_host, 0, 0],
            [moran_host, 1E-2, 0, 0],
            [0, 0, 1E-2, moran_parasitoid],
            [0, 0, moran_parasitoid, 1E-2]
        ] for moran_parasitoid in [0, 1E-4]
    ] for moran_host in [0, 1E-4]
])

nfreqs = 4000
freqs = np.linspace(0, 0.5, nfreqs)
nvars = len(model.vars)

# 2. Compute the spectra for each combination of migration, moran in host and
# parasitoid.
spectra = np.zeros((2, 2, 2, 2, nvars, nvars, nfreqs))
oscfreqs = np.zeros((2, 2, 2, 2, 4))
oscmags = np.zeros((2, 2, 2, 2, 4))

for migh, migp, morh, morp in itertools.product(range(2), repeat=4):
    sym_params = {
        models.Parasitism.params[name]: value
        for name, value in params[migh, migp].iteritems()
    }

    spectra[morh, morp, migh, migp] = np.array([
        model.calculate_spectrum(sym_params, covariance[morh, morp], v)
        for v in freqs
    ]).T

    # TODO: @ Magnitudes are not scaled properly. Also, to what do the non-
    # TODO:     dominant eigenvalues correspond?
    eigvals, eigvecs = model.calculate_eigenvalues(sym_params)
    oscfreqs[morh, morp, migh, migp] = np.angle(eigvals) / (2 * np.pi)
    oscmags[morh, morp, migh, migp] = abs(eigvals)

    print morh, morp, migh, migp, oscfreqs[morh, morp, migh, migp]
    print morh, morp, migh, migp, oscmags[morh, morp, migh, migp]

magspecall = abs(spectra[1, 1, 1, 1])

# 3. Plot cross-spectra and fraction of synchrony.
fig, ax = pp.subplots(16, 4, figsize=(8.5, 22))

for i, ctuple in enumerate(itertools.product(range(2), repeat=4)):
    print "Plotting %d: %d %d %d %d" % ((i,) + ctuple)
    magspec = abs(spectra[ctuple])

    for sp, cell, name in zip([0, 2], [(0, 1), (2, 3)], ['H', 'P']):
        # sp: index of first subplot column. 0 for host, 2 for parasitoid.
        # cell: the cross-spectrum cell index in the full spectral matrix.

        # Plot magnitude spectrum.
        ax[i, sp].plot(freqs, magspec[cell], color='r')
        ax[i, sp].plot(freqs, magspecall[cell], color='k')
        ax[i, sp].set_xlim(freqs[0], freqs[-1])
        ax[i, sp].set_yscale('log')

        ylabel = '$|f_{%s_1%s_2}|$' % (name, name)
        if sp == 0:
            ylabel = 'C=(%d,%d,%d,%d)' % ctuple + ylabel

        ax[i, sp].set_ylabel(ylabel)
        ax[i, sp].set_xlabel('$\omega$')

        # Plot fraction of synchrony.
        ax[i, sp+1].plot(
            freqs, magspec[cell] / magspecall[cell] * 100, color='r'
        )
        ax[i, sp+1].set_xlim(freqs[0], freqs[-1])
        ax[i, sp+1].set_ylabel('$\\%% |f_{%s_1%s_2}|$' % (name, name))
        ax[i, sp+1].set_ylim(0, 110)
        ax[i, sp+1].set_xlabel('$\omega$')

        # Plot horizontal lines for maximum fraction and mean fraction.
        ax[i, sp+1].axhline(
            np.amax(magspec[cell]) / np.amax(magspecall[cell]) * 100,
            color='orange'
        )

        ax[i, sp+1].axhline(
            np.mean(magspec[cell]) / np.mean(magspecall[cell]) * 100,
            color='purple'
        )

        # Plot vertical line at the dominant frequency in both subplots.
        for mag, freq in itertools.izip(oscmags[ctuple], oscfreqs[ctuple]):
            ax[i, sp].axvline(freq, color='green')
            ax[i, sp+1].axvline(freq, color='green')

        # Plot shaded region for 95% of synchrony.
        perc = np.percentile(magspec[cell], 95)

        ax[i, sp].fill_between(
            freqs,
            np.ones(magspec[cell].shape) * np.finfo(magspec[cell].dtype).eps,
            magspec[cell],
            where=magspec[cell] > perc,
            color=(1, 0, 0, .25)
        )

# Titles.
ax[0, 0].set_title('H-H magnitude')
ax[0, 1].set_title('H-H fraction of synchrony')
ax[0, 2].set_title('P-P magnitude')
ax[0, 3].set_title('P-P fraction of synchrony')

# Format "special" rows, i.e. rows for which only one source of synchrony is
# present and models for which both sources are present for either the host or
# the parasitoid, exclusively.
plottitles = {
    (1, 0, 0, 0): 'Host migration',
    (0, 1, 0, 0): 'Parasitoid migration',
    (0, 0, 1, 0): 'Host Moran',
    (0, 0, 0, 1): 'Parasitoid Moran',
    (1, 0, 1, 0): 'Host both',
    (0, 1, 0, 1): 'Parasitoid both'
}

# Red for models where both sources are present in one of the species.
# Blue for models where only one source is present.
for i, ctuple in enumerate(itertools.product(range(2), repeat=4)):
    for sp in range(4):
        if ctuple in plottitles:
            if ctuple in [(1, 0, 1, 0), (0, 1, 0, 1)]:
                ax[i, sp].set_axis_bgcolor((1.0, 0.9, 0.9))
            else:
                ax[i, sp].set_axis_bgcolor((0.9, 0.9, 1.0))

            ax[i, 0].set_title(plottitles[ctuple])

# Save the figure.
pp.subplots_adjust(wspace=0.4, hspace=0.8, top=0.95, bottom=0.05)
pp.savefig('fraction.pdf')

