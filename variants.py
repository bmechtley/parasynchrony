"""
variants.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Plot analytic cross-spectra for a variety of model variants specified by a
pybatchdict JSON config file.
"""

import sys
import os.path
import itertools

import numpy as np

import matplotlib

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=6)
matplotlib.rc('lines', linewidth=0.5)
matplotlib.rc('axes', linewidth=0.5)
matplotlib.rc('font', family='serif')

import matplotlib.pyplot as pp
import matplotlib.cm as cm

import pybatchdict as pbdict

import utilities
import models


def plot(configfile, separate=False):
    # Load the configurations.
    """
    Plot the cospectra for each model variant.

    :param configfile: (str) path to configuration file.
    :param separate: (bool) whether or not to plot each variant in its own plot.
    """

    configdir = os.path.dirname(configfile)
    configs = utilities.load_config(configfile)
    cnames = configs.hyphenate_changes()

    # Cache models if they are used more than once.
    modelcache = {}
    axpanes = None
    ax = None

    # Compute and plot each spectrum.
    for r, (config, cname) in enumerate(itertools.izip(configs.combos, cnames)):
        print 'Plotting %s.' % cname

        # Set up the figure. Outside axes are for the legend.
        if separate or r == 0:
            axpanes = None
            pp.figure(figsize=(15, 15))

            ax = pp.subplot(111)
            ax.patch.set_visible(False)
            pp.axis('off')

        # Fetch the model. If it's already been created, retrieve it from the
        # cache.
        modelname = pbdict.getkeypath(config, '/model/name', 'nbd2')

        if modelname in modelcache:
            model = modelcache[modelname]
        else:
            model = models.parasitism.get_model(modelname)

        # Set up model parameters.
        sym_params = {
            models.parasitism.params[name]: value for name, value in
            pbdict.getkeypath(config, '/model/params').iteritems()
        }

        noise = np.array(pbdict.getkeypath(config, '/model/noise'))

        # Get the spectrum.
        freqs = np.linspace(0, 0.5, config.get('nfreqs', 100))

        spectrum = np.array([
            model.calculate_spectrum(sym_params, noise, v)
            for v in freqs
        ]).T

        # Plot the spectrum.
        plotargs = dict(
            varnames=None if r < len(cnames) - 1 else model.vars,
            plotfun=pp.scatter,
            s=2,
            alpha=2**-5,
            axpanes=axpanes
        )

        if 'color' in config:
            plotargs['color'] = config['color']
        elif separate:
            plotargs['color'] = 'k'
        else:
            plotargs['color'] = cm.rainbow(float(r) / (len(cnames) - 1))

        axpanes = models.plotting.plot_cospectra(freqs, spectrum, **plotargs)

        if separate:
            figpath = os.path.join(configdir, 'spectra-%s.png' % cname)
            print 'Writing %s.' % figpath
            pp.savefig(figpath)

    if not separate:
        # Make a legend for each model using its hyphenated configuration name.
        ax.legend(axpanes[0][0]['top'].get_lines(), cnames)
        # Save the plot.
        figpath = os.path.join(configdir, 'spectra.png')
        print 'Writing %s.' % figpath
        pp.savefig(figpath)


def main():
    """Where the action is."""

    if len(sys.argv) < 1:
        print 'Usage: python variants.py [separate] config.json'
    elif len(sys.argv) > 2:
        plot(sys.argv[2], sys.argv[1] == 'separate')
    else:
        plot(sys.argv[1])

if __name__ == '__main__':
    main()