"""
fraction_host_marginals.py
parasynchrony
2015 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Compute marginal distribution of fraction of synchrony values across the entire
parameter space, including noise parameters.

TODO: Needs some major cleaning up to make this readable re: how the marginals
are organized after processing.
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
    return '\t' * n

def since(btime):
    return time.clock() - btime

def halfshift(a):
    da = np.diff(a)[-1]
    return np.array(list(a) + [a[-1] + da]) - da / 2.

def noise_cov(ndict):
    """
    Noise covariance from parameterization dictionary.

    :param ndict (dict): dictionary of noise parameters including:
        SpSh: Ratio of parasitoid variance to host variance (Sp / Sh).
        Chh: Correlation between hosts.
        Cpp: Correlation between parasitoids.
    :return (np.ndarray): 4x4 covariance matrix.
    """

    return np.array([
        [1, ndict['Chh'], 0, 0],
        [ndict['Chh'], 1, 0, 0],
        [0, 0, ndict['SpSh'], ndict['SpSh'] * ndict['Cpp']],
        [0, 0, ndict['SpSh'] * ndict['Cpp'], ndict['SpSh']]
    ])

def compute_metrics(params):
    """
    Compute the fraction of synchrony between patches as different metrics.

    :param params (dict): model/noise parameter dictionary.
    :return (dict): nested dictionary of modeltype->metric->fraction values.
    """

    metrics = dict()

    for modeltype in ['h', 'p']:
        pden, nden = utilities.dict_split(params, ['SpSh', 'Chh', 'Cpp'])
        pnum, nnum = utilities.dict_split(params, ['SpSh', 'Chh', 'Cpp'])
        pnum['m{0}'.format(modeltype)] = 0
        nnum['C{0}{0}'.format(modeltype)] = 0

        cnum, cden = tuple([
            models.utilities.correlation(model.calculate_covariance(
                models.parasitism.sym_params(p), noise_cov(n)
            ))
            for p, n in [(pnum, nnum), (pden, nden)]
        ])
        cfrac = abs(cnum) / abs(cden)

        metrics[modeltype] = dict(Rhh=cfrac[0, 1], Rpp=cfrac[2, 3])

    return metrics

def compute_marginal(opts):
    params, vkey, vval, vind, histprops, cacheprefix = [opts[k] for k in [
        'params', 'varkey', 'varval', 'varind', 'histprops', 'cacheprefix'
    ]]
    bmin, bmax, bres = [histprops[k] for k in ['min', 'max', 'res']]

    cachepath = '%s-%s-%d-part.pickle' % (cacheprefix, vkey, vind)

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

        counts = dict()

        # Now loop through every combination of values for the embedded
        # hyperplane.
        for vpk in itertools.product(*indices):
            paramset = dict(defaults)

            for i, k in enumerate(varparams):
                paramset[k] = ranges[i][vpk[i]]

            paramset[vkey] = vval

            metrics = compute_metrics(paramset)

            # Loop through to calculate each metric for the point.
            for modeltype, modelmetrics in metrics.iteritems():
                for metric, mval in modelmetrics.iteritems():
                    if modeltype not in counts:
                        counts[modeltype] = dict()

                    if metric not in counts[modeltype]:
                        counts[modeltype][metric] = np.zeros(
                            (len(varparams) + 1, len(ranges[0]), bres)
                        )

                    # TODO: Note: values outside the maximum fraction of
                    # TODO:     synchrony bin (e.g. 10) will be ignored, given
                    # TODO:     NaN values. It may be appropriate to instead
                    # TODO:     stick these in an "overflow" bin or choose a
                    # TODO:     different value.

                    index_y = int(np.interp(
                        mval, [bmin, bmax], [0, bres - 1], right=np.nan
                    ))

                    try:
                        counts[modeltype][metric][0, vind, index_y] += 1
                        for vari in range(len(ranges)):
                            counts[modeltype][metric][
                                vari+1, vpk[vari], index_y
                            ] += 1
                    except IndexError:
                        warnings.warn(
                            '%s %s (%.2f) outside bounds (%s). Params: %s.' % (
                                modeltype, metric, mval,
                                '%.2f-%.2f' % (bmin, bmax),
                                printer.pformat(paramset)
                            )
                        )
                    except ValueError:
                        # warnings.warn(
                        #     'Erroneous %s %s (%.2f). Params: %s.' % (
                        #         modeltype, metric, mval, str(paramset)
                        #     )
                        # )
                        continue

        print tabs(4), 'Finished marginals for %s (%.3fs).' % (
            cachepath, since(tic)
        )
        tic = time.clock()
        cPickle.dump(counts, open(cachepath, 'w'))
        print tabs(4), 'Saved %s (%.3fs).' % (cachepath, since(tic))
    else:
        print tabs(4), 'Loading marginals for %s.' % cachepath
        tic = time.clock()
        counts = cPickle.load(open(cachepath))
        print tabs(4), 'Finished loading %s (%.3fs).' % (cachepath, since(tic))

    return counts

def make_products(config, cacheprefix=''):
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

        # Make a dict modeltype->metric->(nvars, res, bins) and split it up so
        # it is a dict varkey->modeltype->metric->(res, bins)
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
    print 'Plotting %s.' % plotpath
    tic = time.clock()

    marginals = products['marginals']
    config = products['config']
    hist = config['args']['histogram']
    bins = np.linspace(hist['min'], hist['max'], hist['res'])

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
        pp.contourf(
            param['range'], bins, zs,
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            levels=10**np.linspace(vlogmin, vlogmax, 256)
        )

        pp.xlabel('$%s$' % models.parasitism.symbols[key])
        pp.ylabel('$R_{hh}$')
        pp.xlim(np.amin(param['range']), np.amax(param['range']))
        pp.ylim(np.amin(bins), np.amax(bins))
        pp.xticks(param['range'], ['%.2f' % p for p in param['range']])

        cb = pp.colorbar(ticks=10**np.arange(int(vlogmin), int(vlogmax) + 1))
        cb.set_label(r'\# samples')

    pp.subplots_adjust(top=0.975, bottom=0.025)
    pp.savefig(plotpath)
    print tabs(4), 'Saved %s (%.3fs).' % (plotpath, since(tic))

def main():
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

        configdir, configname = 'cache/', 'fraction-host-marginals-default'

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
