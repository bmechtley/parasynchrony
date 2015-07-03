"""
marginals_bivariate.py
parasynchrony
2015 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Usage: python marginals_bivariate.py configname.json
    configname.json: configuration file describing model parameters and ranges,
        computation resolution, number of processes used, and plotting
        preferences.

Produces intermediate cached pickle files in the same path as the config
file, defaulting to cache/. Saves a plot in plots/configname.png.

The input configuration JSON should be formatted as follows:
{
    "args": {
        "resolution": /* number of parameter values for each parameter */,
        "processes": /* number of processes to use for parallel computation */,
        "histogram": {
            "min": /* minimum fraction of synchrony to bin */,
            "max": /* maximum fraction of synchrony to bin */,
            "res": /* number of histogram bins */
        },
    }
    "params": {
        [paramname]: {
            "default": /* default value for parameter. Shouldn't actually be
                used by this script. */,
            "range": (min, max) /* where min, max or minimum and maximum values
                for the parameter */
        }
    }
}
"""

import os
import sys
import time
import json
import glob
import pprint
import cPickle
import warnings
import itertools
import multiprocessing

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as pp

import models
import utilities

model = models.parasitism.get_model('nbd(2)')
model.lambdify_ss = False
printer = pprint.PrettyPrinter()


def tabs(n):
    """
    Return a string containing n tabs. Helper for debug output Laaaaazy.

    :param n: number of tab characters to return.
    :return: a string containing n tab characters.
    """

    return '\t' * n


def since(btime):
    """
    Return time since the input time.

    :param btime: (float) current processor time (presumably from time.clock())
    :return: (float) time since input time in milliseconds
    """
    return time.clock() - btime


def noise_cov(ndict):
    """
    Noise covariance from parameterization dictionary.

    :param ndict: (dict) dictionary of noise parameters including:
        SpSh: Ratio of parasitoid variance to host variance (Sp / Sh).
        Chh: Correlation between hosts.
        Cpp: Correlation between parasitoids.
    :return: (np.ndarray) 4x4 covariance matrix.
    """

    return np.array([
        [1, ndict['Chh'], 0, 0],
        [ndict['Chh'], 1, 0, 0],
        [0, 0, ndict['SpSh'], ndict['SpSh'] * ndict['Cpp']],
        [0, 0, ndict['SpSh'] * ndict['Cpp'], ndict['SpSh']]
    ])


def compute_metrics(params):
    """
    Compute the fracsync between patches as different metrics. Fracsync is
    defined as a division of two correlation matrices. Depending on which sync.
    effects are being tested (host or para.), the numerator is the correlation
    matrix for the model having ONLY sync. effects for the pop. in question, and
    the denominator is the full model with sync. effects on both pops.

    Note that this "fraction" can go above 1, in which case the sync. effects on
    the OTHER pop. are actually de-synchronizing influences.

    :param params: (dict) model / noise parameters. See the main() docstring for
        how this dictionary should be formatted.
    :return: (dict) Fracsync values. Layout:
        h and p: (dict) For each sync. effects on hosts (h) and paras. (p).
            Rhh and Rpp: (float) Fracsync of hosts and fracsync of paras.
    """

    metrics = dict()

    for effects in ['h', 'p']:
        pden, nden = utilities.dict_split(params, ['SpSh', 'Chh', 'Cpp'])
        pnum, nnum = utilities.dict_split(params, ['SpSh', 'Chh', 'Cpp'])
        pnum['m{0}'.format(effects)] = 0
        nnum['C{0}{0}'.format(effects)] = 0

        cnum, cden = tuple([
            models.utilities.correlation(model.calculate_covariance(
                models.parasitism.sym_params(p), noise_cov(n)
            ))
            for p, n in [(pnum, nnum), (pden, nden)]
        ])
        cfrac = abs(cnum) / abs(cden)

        metrics[effects] = dict(Rhh=cfrac[0, 1], Rpp=cfrac[2, 3])

    return metrics


def compute_marginal(opts):
    """
    Compute bivariate histogram marginals for a single value of a given
    parameter For example, if the parameter is "host correlation," assign some
    constant value (e.g. 0.5) to it and compute fracsync metrics for every
    combination of other parameters, binning the resulting values in histograms,
    computing the distribution of fracsync values.

    This will save an intermediate cached pickle file which can then opened by
    make_products to combine all the histograms into 2D marginals, varying the
    values for the parameter in question. 1D histograms are cached so that the
    process can be interrupted and continued without losing progress.

    :param opts: (dict) dictionary of options passed in from an element in a
        list of parameters given to multiprocessing.Pool. Layout:
            params: (dict) parameters. See main() docstring for info on how this
                is formatted. Parameter ranges are replaced with (res,) numpy
                arrays made using np.linspace.
            varkey: (str) which parameter is varying for this run.
            varval: (float) the value varkey should take on.
            varind: (int) the index of the value in the parameter's range of
                values.
            histprops: (dict) properties for the histogram. Layout:
                    min: (float) minimum fracsync to bin
                    max: (float) maximum fracsync to bin
                        NOTE: values that fall outside the max are all gathered
                        in the top bin.
                    res: (float) number of bins for histogram.
                A fracsync value's bin will be computed by linearly
                interpolating min->max between 0->res.
    :return: (dict) Histograms of fraction of synchrony values. Layout:
            h and p: (dict) keyed by effect for testing host (h) or para. (p)
                sync. effects.
                Rhh and Rpp: (np.ndarray) 1D histogram of fracsync values on
                hosts (Rhh) or paras. (Rpp).
    """

    params, vkey, vval, vind, histprops, cacheprefix = [opts[k] for k in [
        'params', 'varkey', 'varval', 'varind', 'histprops', 'cacheprefix'
    ]]
    bmin, bmax, bres = [histprops[k] for k in ['min', 'max', 'res']]

    cachepath = '%s-%s-part.pickle' % (cacheprefix, vkey)

    if not os.path.exists(cachepath):
        tic = time.clock()

        defaults = {k: v['default'] for k, v in params.iteritems()}
        varparams = {
            k: v for k, v in params.iteritems()
            if len(v['range']) > 1 and k != vkey
        }
        ranges = [p['range'] for p in varparams.itervalues()]
        indices = [np.arange(len(p)) for p in ranges]
        print 'Computing marginals for %s.' % cachepath

        hist = dict()

        # Now loop through every combination of values for the embedded
        # hyperplane.
        for vpk in itertools.product(*indices):
            paramset = dict(defaults)

            for i, k in enumerate(varparams):
                paramset[k] = ranges[i][vpk[i]]

            paramset[vkey] = vval

            metrics = compute_metrics(paramset)

            # Loop through to calculate each metric for the point.
            for effects, modelmetrics in metrics.iteritems():
                for metric, mval in modelmetrics.iteritems():
                    hist.setdefault(effects, dict())
                    hist[effects].setdefault(metric, np.zeros(
                        (len(varparams) + 1, len(ranges[0]), bres)
                    ))
                    bind = int(np.interp(mval, [bmin, bmax], [0, bres - 1]))

                    try:
                        hist[effects][metric][0, vind, bind] += 1

                        for vari in range(len(ranges)):
                            hist[effects][metric][vari+1, vpk[vari], bind] += 1
                    except IndexError:
                        # This should never happen, since np.interp is clipped.
                        warnings.warn(
                            '%s %s (%.2f) outside bounds (%s). Params: %s.' % (
                                effects, metric, mval,
                                '%.2f-%.2f' % (bmin, bmax),
                                printer.pformat(paramset)
                            )
                        )
                    except ValueError:
                        # NaN / inf values. Shouldn't happen, given proper
                        # parameter ranges.
                        warnings.warn('Erroneous %s %s (%.2f). Params: %s.' % (
                            effects, metric, mval, str(paramset)
                        ))
                        continue

        print tabs(4), 'Finished marginals for %s (%.3fs).' % (
            cachepath, since(tic)
        )
        tic = time.clock()
        cPickle.dump(hist, open(cachepath, 'w'))
        print tabs(4), 'Saved %s (%.3fs).' % (cachepath, since(tic))
    else:
        print tabs(4), 'Loading marginals for %s.' % cachepath
        tic = time.clock()
        hist = cPickle.load(open(cachepath))
        print tabs(4), 'Finished loading %s (%.3fs).' % (cachepath, since(tic))

    return hist


def make_products(config, cacheprefix=''):
    """
    Compute 2D marginal histograms for fraction of synchrony values for
    combinations of all model parameters. Fraction of synchrony is defined in
    terms of fracsync on hosts due to host effects and on parasitoids due to
    parasitoid effects.

    :param config: (dict) configuration dictionary. See main() docstring for
        explanation of how this should be formatted. Note that parameter
        ranges are replaced with np.ndarrays of their values from np.linspace.
    :param cacheprefix: (str) path/filename prefix to be used for intermediate
        cached files for picking up where left off if the processes are
        interrupted. There is a main cache file which is the end, combined
        result of all computations called [cacheprefix]-products.pickle and
        there are intermediate cache files for the 1D marginals for parameter
        value produced by compute_marginals.
    :return: dictionary of 2D marginal histograms. Layout is as follows:
        marginals: (dict)
            [all parameter keys]: (dict)
                h and p: (dict) for each effect, e.g. only host sync. effects
                    or parasitoid sync effects.
                    Rhh and Rpp: (np.ndarray) 2D histogram of fracsync. values
                    for sync. on hosts (Rhh) and paras. (Rpp). Rows correspond
                    values for the parameter and columns correspond to fracsync.
                    histogram bins.
    """

    cachepath = '%s-products.pickle' % cacheprefix

    args = config['args']
    params = config['params']

    if not os.path.exists(cachepath):
        print 'Computing with %d processes.' % config['args']['processes']
        tic = time.clock()

        varparams = {k: v for k, v in params.iteritems() if len(v['range']) > 1}
        varkeys = varparams.keys()
        pivotvar = varkeys[0]
        indices = np.arange(len(varparams[pivotvar]['range']))

        # This processes all the binning by computing each different "pivot"
        # variable in a separate process. TODO: Allow for better parallelism,
        # as this is only going to allow for a number of processes equal to the
        # number of parameters.
        results = utilities.multipool(
            compute_marginal,
            [
                dict(
                    params=config['params'],
                    varkey=pivotvar,
                    varval=varparams[pivotvar]['range'][vi],
                    varind=vi,
                    cacheprefix=cacheprefix,
                    histprops=args['histogram']
                )
                for vi in indices
            ],
            processes=config['args']['processes']
        )

        # Make a dict effects->metric->(nvars, res, bins) and split it up so
        # it is a dict varkey->effects->metric->(res, bins)
        marginals = {
            varkey: {
                mtkey: {
                    metrickey: metricitem[vi]
                    for metrickey, metricitem in mtvalue.iteritems()
                } for mtkey, mtvalue in {
                    k1: {
                        k2: np.sum([r[k1][k2] for r in results], axis=0)
                        for k2 in results[0][k1]
                    }
                    for k1 in results[0]
                }.iteritems()
            } for vi, varkey in enumerate(varkeys)
        }

        # And now hope it all worked.
        products = dict(marginals=marginals, config=config)

        print tabs(4), 'Finished computing metrics (%.3fs).' % since(tic)
        cPickle.dump(products, open(cachepath, 'w'))
        print tabs(4), 'Saved %s.' % cachepath
        for d in glob.glob('%s-*-part.pickle' % cacheprefix):
            os.remove(d)
    else:
        tic = time.clock()
        print tabs(4), 'Loading %s.' % cachepath
        products = cPickle.load(open(cachepath))
        print tabs(4), 'Finished loading (%.3fs).' % since(tic)

    return products


def plot_marginals(products, plotpath):
    """
    For each model parameter, plot 2D marginal histograms with parameter values
    on the X axis and fraction of synchrony values on the Y axis, Z axis (color)
    corresponding to bin counts.

    :param products: (dict) Output from make_products. See its docstring for
        information on how this is organized.
    :param plotpath: (str) where to save the plot.
    """

    print 'Plotting %s.' % plotpath
    tic = time.clock()

    marginals = products['marginals']
    config = products['config']
    hist = config['args']['histogram']
    bins = np.linspace(hist['min'], hist['max'], hist['res'])

    # Size the figure according to how many parameters there are.
    nparams = len(marginals)
    npx = int(nparams ** 0.5)
    npy = (nparams + 1) / npx
    pp.figure(figsize=(npx * 7, npy * 5))

    for i, (key, marginal) in enumerate(marginals.iteritems()):
        param = config['params'][key]
        pp.subplot(npy, npx, i + 1)

        # Contour plot of 2D histogram.
        zs = marginal['h']['Rhh'].T
        vmin, vmax = np.amin(zs[zs != 0]), np.amax(zs)
        vlogmin, vlogmax = np.log10(vmin), np.log10(vmax)

        # TODO: why is PyCharm complaining?
        pp.contourf(
            param['range'], bins, zs,
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            levels=10**np.linspace(vlogmin, vlogmax, 256)
        )

        # Fix axis limits/labels/ticks.
        pp.xlabel('$%s$' % models.parasitism.symbols[key])
        pp.ylabel('$R_{hh}$')
        pp.xlim(np.amin(param['range']), np.amax(param['range']))
        pp.ylim(np.amin(bins), np.amax(bins))
        pp.xticks(param['range'], ['%.2f' % p for p in param['range']])

        # Make colorbar.
        cb = pp.colorbar(ticks=10**np.arange(int(vlogmin), int(vlogmax) + 1))
        cb.set_label(r'\# samples')

    pp.subplots_adjust(top=0.975, bottom=0.025)
    pp.savefig(plotpath)
    print tabs(4), 'Saved %s (%.3fs).' % (plotpath, since(tic))


def main():
    """Where the action is."""

    if len(sys.argv) > 1:
        configpath = sys.argv[1]
        configdir, configfile = os.path.split(configpath)
        configname = os.path.splitext(configfile)[0]
        config = json.load(open(configpath))
    else:
        config = dict(
            args=dict(
                resolution=10,
                processes=multiprocessing.cpu_count() - 1,
                histogram=dict(min=0, max=10, res=100)
            ),
            params=dict(
                r=dict(default=2.0, range=[1.1, 4]),
                a=dict(default=1.0),
                c=dict(default=1.0),
                k=dict(default=0.5, range=[0.1, 0.9]),
                mh=dict(default=0.25, range=[0, 0.5]),
                mp=dict(default=0.25, range=[0, 0.5]),
                SpSh=dict(default=1.0, range=[0, 10]),
                Chh=dict(default=0.5, range=[0, 1]),
                Cpp=dict(default=0.5, range=[0, 1])
            )
        )

        configdir, configname = 'cache/', 'fracsync-marginals-default'

    ncpus, res = config['args']['processes'], config['args']['resolution']

    for p in config['params'].itervalues():
        p.setdefault('res', res)
        p.setdefault('range', (p['default'], p['default']))

        if p['range'][0] != p['range'][1]:
            p['range'] = np.linspace(p['range'][0], p['range'][1], res)
        else:
            p['range'] = [p['default']]

    products = make_products(config, os.path.join(configdir, configname))

    if config.get('args', {}).get('plot', False):
        plot_marginals(products, os.path.join('plots', '%s.png' % configname))

if __name__ == '__main__':
    main()
