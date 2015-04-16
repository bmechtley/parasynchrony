"""
fraction_host.py
2015 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Plot fraction of synchrony in the host that is a result of host synchronizing
effects (migration, Moran). Fraction of synchrony in this case is defined as
the Pearson correlation coefficient between the two host populations in either
patch without any parasitoid synchronizing influences (i.e. Spp=0, mp=0) divided
by the correlation between the two hosts with all synchronizing influences.

TODO: Change color scheme so the curve < 1 is easy to grasp (i.e. is it
TODO:   logarithmic? - perhaps use banding) and values > 1 are still
TODO:   distinguishable (e.g. fade to white). Put all plots on the same
TODO:   cmap/norm.
ope
TODO: Run at higher resolutions.
TODO: Identify interesting combinations/regions.
TODO: Try to find a reasonable bounding box over which to average these things.

TODO: Run with origin within weird k/r / k/mp region.
"""

import os
import json
import cPickle
import argparse

from itertools import combinations_with_replacement, izip, chain

import collections
import multiprocessing

import numpy as np
import scipy.stats
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as pp

import models

model = models.parasitism.get_model('nbd(2)')


def poolmap(pool, *args):
    """
    Simple helper to make a parallel processing pool respond to
    KeyboardInterrupt.
    :param pool (multiprocessing.Pool): the pool.
    :param args: positional arguments for pool.map_async(...)
    :return (list): mapped results.
    """

    return pool.map_async(*args).get(999999)


def fraction_synchrony(params, nfreqs=100):
    """
    Compute the fraction of synchrony between patches for which the host is
    responsible using different metrics.

    TODO: Allow caching of spectra.
    TODO: Return fraction for which different populations are responsible.
    TODO: Optimize.
    TODO: Allow for different parameter ranges.

    :param params: model parameter dictionary. Contains "num" and "den" keys
        that contain parameter keys.
    :param nfreqs: number of frequency values to compute.
    :return: dictionary of metric: fraction pairs.
    """

    fracfuns = dict(
        fracsync=lambda n, d: np.divide(
            np.mean(np.abs(n), axis=0), np.mean(np.abs(d), axis=0)
        ),
        corrnum=lambda n, d: np.mean(np.abs(n), axis=0),
        corrden=lambda n, d: np.mean(np.abs(d), axis=0)
    )

    freqs = np.linspace(0, 0.5, nfreqs)

    corr, cov = dict(), dict()

    for name in ['num', 'den']:
        newparams = {k: v for k, v in params[name].iteritems() if k not in [
            'Spp', 'Sp', 'Sh', 'Shh'
        ]}

        sym_params = models.parasitism.sym_params(newparams)

        noisecov = np.array([
            [params[name]['Sh'], params[name]['Shh'], 0, 0],
            [params[name]['Shh'], params[name]['Sh'], 0, 0],
            [0, 0, params[name]['Sp'], params[name]['Spp']],
            [0, 0, params[name]['Spp'], params[name]['Sp']]
        ])

        spectrum = np.abs(np.array([
            model.calculate_spectrum(sym_params, noisecov, f) for f in freqs
        ]))

        cov[name] = model.calculate_covariance(sym_params, noisecov)

        # Compute normalized spectrum, i.e. correlation matrix.
        corr[name] = np.rollaxis(np.array([
            [
                np.divide(
                    np.abs(spectrum[:, row, col]),
                    np.sqrt(np.prod([
                        cov[name][pop, pop]**2 for pop in [row, col]
                    ]))
                ) for col in range(cov[name].shape[1])
            ] for row in range(cov[name].shape[0])
        ]), 2, 0)

    results = {
        k: fracfun(corr['num'], corr['den'])
        for k, fracfun in fracfuns.iteritems()
    }

    # results['correlations'] = corr
    # results['covariances'] = cov

    return results


def dict_merge(a, b):
    """
    Merge two dictionaries, preferring the values from the second in case of
    collision.

    :param a: first dictionary
    :param b: second dictionary
    :return: new dictionary containing keys and values from both dictionaries.
    """

    c = a.copy()
    c.update(b)
    return c


def process_products(opts):
    """
    Parallel worker for computing fraction of average synchrony. Used by
    make_products.

    :param opts (dict): dictionary of input parameters, including keys:
        params (dict): parameter dictionary of the form
            {name: {default: default, range: (low, high), res: resolution}.
        k1 (str): Y parameter name.
        k2 (str): X parameter name.
    :return: (R1, R2) dimensional list (not array) of dictionaries where each
        key is a different synchrony metric, e.g. fracavgsync. See
        fraction_synchrony for more info.
    """

    varyingkeys = [k for k in opts['params'] if opts['params'][k]['res'] > 1]
    defaults = {k: v['default'] for k, v in opts['params'].iteritems()}
    k1, k2 = opts['k1'], opts['k2']
    r1, r2 = [opts['params'][k]['range'] for k in [k1, k2]]

    keycombos = list(combinations_with_replacement(varyingkeys, 2))
    strargs = (keycombos.index((k1, k2)) + 1, len(keycombos), k1, k2)
    print 'Processing %d / %d (%s, %s).' % strargs

    fracsync = lambda a, b: fraction_synchrony(dict(
        num=dict_merge(defaults, {k1: a, k2: b, 'Spp': 0, 'mp': 0}),
        den=dict_merge(defaults, {k1: a, k2: b})
    ))

    if k1 != k2:
        result = [[fracsync(v1, v2) for v2 in r2] for v1 in r1]
        result = {
            k: np.array([
                [
                    cell[k] for cell in row
                ] for row in result
            ])
            for k in result[0][0].keys()
        }
    else:
        result = [fracsync(v1, v1) for v1 in r1]
        result = {
            k: np.array([cell[k] for cell in result])
            for k in result[0].keys()
        }

    print '\t\t\t\t\tCompleted %d / %d (%s, %s).' % strargs
    return result


def make_products(params=None, pool=None):
    """
    Compute fraction of synchrony metrics across combinations of values for each
    pair of model parameters.

    :param params (dict): Parameter dictionary, of the form
        {name: {default: default, range: (low, high), res: resolution}.
    :param pool (multiprocessing.Pool): pool used for parallel computation.
    :return: (P, P, R1, R2) dimensional list (not array) of dictionaries where
        each key is a different synchrony metric, e.g. fracavgsync. See
        fraction_synchrony for more info.
    """

    varyingkeys = [k for k in params if params[k]['res'] > 1]
    print 'Varying keys:', varyingkeys

    keyproduct = list(combinations_with_replacement(varyingkeys, 2))

    # Returns an array of dictionaries, each containing a two-dimensional array
    # with the different metrics returned by fraction_synchrony across the
    # combination of each pair of varying parameters.
    products = poolmap(pool, process_products, [
        dict(params=params, k1=k1, k2=k2) for k1, k2 in keyproduct
    ])

    # Restructure so we have a dictionary with fraction_synchrony metrics as
    # keys and lists of two-dimensional arrays of their values across the
    # combination of each pair of varying parameters.
    products = {k: [p[k] for p in products] for k in products[0].keys()}

    # Finally, return a dict such that each fraction_synchrony metric key is
    # associated with a (K, K, N, N) array, where K is the number of varying
    # keys and N is the parameter resolution.
    return {
        k: [
            [
                [
                    x for i, x in enumerate(v)
                    if (keyproduct[i][0] == k1 and keyproduct[i][1] == k2)
                    or (keyproduct[i][0] == k2 and keyproduct[i][1] == k1)
                ][0]
                for k2 in varyingkeys
            ] for k1 in varyingkeys
        ]
        for k, v in products.iteritems()
    }


def plot_fracsync(params=None, metric=None, filename=None, metricname=""):
    """
    Plot fraction of host synchrony due to host effects across combinations of
    values for each pair of model parameters.

    :param params (dict): Parameter dictionary, of the form
        {name: {default: default, range: (low, high), res: resolution}. Should
        be the same dictionary used to produce the products parameter.
    :param products (list): (P, P, R1, R2) dimensional list (not array)
        containing dictionaries where each key is a different synchrony metric,
        e.g. fracavgsync. See fraction_synchrony for more info.
    :param filename (str): Output filename for plot.
    """

    # Get fraction of host synchrony from host effects for all combinations
    # parameters that vary over more than one value (res > 1).
    vkeys = [k for k in params if params[k]['res'] > 1]
    n = len(vkeys)

    # Get minimum and maximum values.
    allvals = np.concatenate([
        np.concatenate([
            metric[i][j].flatten() for j in range(len(metric[i]))
        ]).flatten() for i in range(len(metric))
    ]).flatten()

    minval, maxval = np.amin(allvals), np.amax(allvals)
    valrange = maxval - minval

    # Make a new colormap with one colormap for values 0-1.0 fraction of sync.
    # and another colormap for values 1.0+ fraction of sync.
    ncolors = 1024
    cmap_low = matplotlib.cm.get_cmap('gnuplot2')
    cmap_high = matplotlib.cm.get_cmap('Greens_r')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'newcmap', [(i, (c[0], c[1], c[2], c[3])) for i, c in izip(
            chain(
                np.linspace(0, 1 / valrange, ncolors, endpoint=False),
                np.linspace(1 / valrange, 1, ncolors),
            ),
            chain(
                [cmap_low(x) for x in np.linspace(0, 1, ncolors)],
                [cmap_high(x) for x in np.linspace(1, .5, ncolors)]
            )
        )]
    )

    fig, subplots = pp.subplots(n, n, figsize=(5 * n + 5, 3.75 * n + 5))
    fig.suptitle('Frac. avg. sync. no para moran/dispersal / all effects')

    # i is row (y axis), j is column (x axis)
    for (i, ki, si, li), (j, kj, sj, lj) in combinations_with_replacement(
        zip(
            range(len(vkeys)),
            vkeys,
            [models.parasitism.symbols[k] for k in vkeys],
            [models.parasitism.labels[k] for k in vkeys]
        ),
        2
    ):
            ri, rj = params[ki]['range'], params[kj]['range']
            di, dj = params[ki]['default'], params[kj]['default']

            # If there is only one varying key, subplots(1, 1, ...) returns
            # just the first axes rather than a multidimensional list.
            try:
                ax = subplots[n - i - 1][n - j - 1]
            except TypeError:
                ax = subplots

            if i != j:
                subplots[i][j].set_axis_off()

                # 1. Plot fraction of average synchrony.
                imshow = ax.imshow(
                    metric[i][j],
                    vmin=minval,
                    vmax=maxval,
                    cmap=cmap,
                    label='host',
                    aspect='auto',
                    extent=(rj[0], rj[-1], ri[-1], ri[0])
                )

                flat = metric[i][j].flatten()
                percs = [25, 50, 75]
                percvals = [np.percentile(flat, q) for q in percs]

                # 2. Plot contours.
                contour = ax.contour(
                    rj, ri, metric[i][j],
                    colors=['w', 'w', 'w'],
                    levels=list(percvals),
                    linestyles=['dotted', 'dashed', 'solid']
                )

                # 3. Axes.
                ax.set_xlabel('$%s$ (%s)' % (sj, lj))
                ax.set_ylabel('$%s$ (%s)' % (si, li))
                ax.axvline(dj, color='pink', ls=':', lw=2)
                ax.axhline(di, color='pink', ls=':', lw=2)
                ax.set_xlim(rj[0], rj[-1])
                ax.set_ylim(ri[0], ri[-1])
                ax.tick_params(axis='both', which='major', labelsize=8)

                # 4. Labels.
                colorbar = pp.colorbar(imshow, ax=ax)
                colorbar.ax.tick_params(labelsize=8)
                ax.clabel(contour, inline=1, fontsize=8)
            else:
                ax.plot(ri, metric[i][j], label=metricname)

                lineprops = dict(color='r', ls=':')
                ax.axvline(di, label='origin', **lineprops)
                ax.axhline(np.interp(di, ri, metric[i][j]), **lineprops)

                ax.set_xlabel('$%s$ (%s)' % (si, li))
                ax.set_ylabel(metricname)
                ax.set_xlim(min(ri), max(ri))
                ax.set_ylim(min(metric[i][j]), max(metric[i][j]))
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.legend()

    pp.subplots_adjust(
        hspace=0.4, wspace=0.4,
        left=0.05, bottom=0.05,
        top=0.95, right=0.95
    )

    pp.savefig(filename, dpi=240)


def main():
    parser = argparse.ArgumentParser(
        description='Fraction of synchrony in host due to host effects.'
    )
    parser.add_argument(
        '-np', '--poolcpus',
        type=int,
        default=multiprocessing.cpu_count() - 1,
        help='how many processes to run concurrently'
    )
    parser.add_argument(
        '-r', '--resolution',
        type=int,
        default=10,
        help='number of steps in each dimension'
    )
    args = parser.parse_args()

    params = collections.OrderedDict(
        r=dict(default=2.0, range=(1.1, 4.0), res=1),
        a=dict(default=1.0, res=1),
        c=dict(default=1.0, res=1),
        k=dict(default=0.16, range=(0.1, 0.25)),
        mh=dict(default=0.25, range=(0.125, 0.5)),
        mp=dict(default=0.25, range=(0.125, 0.5)),
        Sh=dict(default=0.5, range=(0.125, 1.0)),
        Shh=dict(default=0.25, range=(0.125, 1.0)),
        Sp=dict(default=0.5, range=(0.125, 1.0)),
        Spp=dict(default=0.25, range=(0.125, 1.0))
    )

    # params = collections.OrderedDict(
    #     r=dict(default=2, res=1),
    #     a=dict(default=1, res=1),
    #     c=dict(default=1, res=1),
    #     k=dict(default=.5, range=(.1, .25)),
    #     mh=dict(default=.25, res=1),
    #     mp=dict(default=.25, res=1),
    #     Sh=dict(default=.5, res=1),
    #     Shh=dict(default=.25, res=1),
    #     Sp=dict(default=.5, res=1),
    #     Spp=dict(default=.25, res=1)
    # )

    phash = hash(json.dumps(dict_merge(params, vars(args))))

    for p in params.itervalues():
        p.setdefault('res', args.resolution)
        p.setdefault('range', (p['default'], p['default']))
        p['range'] = np.linspace(p['range'][0], p['range'][1], p['res'])

    pool = multiprocessing.Pool(processes=args.poolcpus)

    cachepath = 'cache/fraction-host-%d-%s.pickle' % (args.resolution, phash)
    plotprefix = 'plots/fraction-host-%d-%s' % (args.resolution, phash)

    for dirname in [os.path.dirname(path) for path in [cachepath, plotprefix]]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    if not os.path.exists(cachepath):
        print 'Computing %s with %d processes.' % (cachepath, args.poolcpus)
        products = make_products(params=params, pool=pool)

        print 'Writing %s.' % cachepath
        cPickle.dump(products, open(cachepath, 'w'))
    else:
        print 'Loading %s.' % cachepath
        products = cPickle.load(open(cachepath))

    metrics = {
        k: [
            [
                pj[:, :, 0, 1]
                if pj.ndim == 4
                else pj[:, 0, 1]
                for pj in pi
            ] for pi in v
        ]
        for k, v in products.iteritems()
    }

    for k, v in metrics.iteritems():
        plotpath = '%s-%s.png' % (plotprefix, k)
        print 'Plotting %s.' % plotpath
        plot_fracsync(params, v, plotpath, k)

if __name__ == '__main__':
    main()
