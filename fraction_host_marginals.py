import os
import sys
import time
import json
import glob
import cPickle
import itertools
import multiprocessing

import numpy as np

import models
import utilities

model = models.parasitism.get_model('nbd(2)')


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
        pnum['m{0}' % modeltype] = 0
        nnum['C{0}{0}' % modeltype] = 0

        cnum, cden = tuple([
            models.utilities.correlation(model.calculate_covariance(
                models.parasitism.sym_params(p), noise_cov(n)
            ))
            for p, n in [(pnum, nnum), (pden, nden)]
        ])

        cfrac = cnum / cden

        metrics[modeltype] = dict(Rhh=cfrac[0, 1], Rpp=cfrac[2, 3])

    return metrics


def compute_marginal(opts):
    cachepath = '%s-%s-part.pickle' % (opts['cacheprefix'], opts['varkey'])

    if not os.path.exists(cachepath):
        print 'Computing marginals for %s.' % opts['varkey']
        btime = time.clock()

        binmin, binmax, binres = opts['binmin'], opts['binmax'], opts['binres']

        varparams = {
            k: v for k, v in opts['params'].iteritems() if len(v['range']) > 1
        }
        varparams[opts['varkey']] = opts['params'][opts['varkey']]['default']

        defaults = {k: v['default'] for k, v in opts['params'].iteritems()}

        counts = dict()

        for vpk in itertools.product([p['range'] for p in varparams.itervalues()]):
            paramset = dict(defaults)

            for i, k in enumerate(varparams):
                paramset[k] = vpk[i]

            metrics = compute_metrics(paramset)
            for mval, metric in metrics.iteritems():
                if metric not in counts:
                    counts[metric] = np.zeros(opts['binres'])

                try:
                    counts[metric][int((mval - binmin) / (binmax - binmin))] += 1
                except IndexError:
                    continue

        print '\t\t\t\tFinished computing marginals for %s (%.3fs).' % (
            opts['varkey'], time.clock() - btime
        )
        btime = time.clock()
        cPickle.dump(counts, open(cachepath, 'w'))
        print '\t\t\t\tSaved %s (%.3fs).' % (cachepath, time.clock() - btime)
    else:
        btime = time.clock()
        print 'Loading marginals for %s.' % cachepath
        counts = cPickle.load(open(cachepath))
        print '\t\t\t\tFinished loading %s (%.3fs).' % (
            cachepath, time.clock() - btime
        )

    return counts


def make_products(
        params,
        ncpus=multiprocessing.cpu_count(),
        cacheprefix='',
        binmin=0,
        binmax=10,
        binres=100
):
    cachepath = '%s-products.pickle' % cacheprefix

    if not os.path.exists(cachepath):
        print 'Computing with %d processes.' % ncpus
        btime = time.clock()

        varparams = {k: v for k, v in params.iteritems() if len(v['range']) > 1}

        products = utilities.imultipool(
            compute_marginal,
            [
                dict(
                    params=params,
                    k=k,
                    binmin=binmin,
                    binmax=binmax,
                    binres=binres,
                    cacheprefix=cacheprefix
                )
                for k in varparams
            ],
            processes=ncpus
        )

        print '\t\t\t\tFinished computing all metrics (%.3fs).' % (time.clock() - btime)
        cPickle.dump(products, open(cachepath, 'w'))
        print '\t\t\t\tSaved %s.' % cachepath
        for d in glob.glob('%s-*-part.pickle' % cacheprefix):
            os.remove(d)
    else:
        btime = time.clock()
        print '\t\t\t\tLoading %s.' % cachepath
        products = cPickle.load(open(cachepath))
        print '\t\t\t\tFinished loading %s.' % (time.clock() - btime)

    return products


def main():
    configpath = sys.argv[1]
    configdir, configfile = os.path.split(configpath)
    configname = os.path.splitext(configfile)[0]
    config = json.load(open(configpath))

    # Params I care about:
    #   Chh, Cpp, SpSh, mh, mp, r, k

    ncpus, res = config['args']['processes'], config['args']['resolution']

    for p in config['params'].itervalues():
        p.setdefault('res', res)
        p.setdefault('range', (p['default'], p['default']))
        p['range'] = np.linspace(p['range'][0], p['range'][1], res)

    results = make_products(
        config['params'],
        ncpus=ncpus,
        cacheprefix=os.path.join(configdir, configname)
    )

    print results


if __name__ == '__main__':
    main()