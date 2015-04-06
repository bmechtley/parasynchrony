import os
import cPickle
import itertools
import collections
import multiprocessing

import numpy as np
import matplotlib.pyplot as pp

import models

model = models.parasitism.get_model('nbd(2)')


def poolmap(pool, *args):
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
        # avgfracsync=lambda n, d: np.mean(n / d, axis=0),
        # maxfracsync=lambda n, d: np.amax(n / d, axis=0),
        fracavgsync=lambda n, d: np.mean(np.abs(n), axis=0) / np.mean(np.abs(d), axis=0)
        # fracmaxsync=lambda n, d: np.amax(n, axis=0) / np.amax(d, axis=0)
    )

    freqs = np.linspace(0, 0.5, nfreqs)
    corr = dict()

    for name in ['num', 'den']:
        newparams = dict(params[name])
        del newparams['Spp']
        del newparams['Sp']
        del newparams['Sh']
        del newparams['Shh']
        sym_params = models.parasitism.sym_params(newparams)

        noisecov = np.array([
            [params[name]['Sh'], params[name]['Shh'], 0, 0],
            [params[name]['Shh'], params[name]['Sh'], 0, 0],
            [0, 0, params[name]['Sp'], params[name]['Spp']],
            [0, 0, params[name]['Spp'], params[name]['Sp']]
        ])

        spectrum = np.abs(np.array([
            model.calculate_spectrum(sym_params, noisecov, freq)
            for freq in freqs
        ]))

        cov = model.calculate_covariance(sym_params, noisecov)

        # Compute normalized spectrum, i.e. correlation matrix.
        corr[name] = np.rollaxis(np.array([
            [
                np.divide(
                    np.real(spectrum[:, row, col]),
                    np.sqrt(np.prod([cov[c, c]**2 for c in [row, col]]))
                ) for col in range(cov.shape[1])
            ] for row in range(cov.shape[0])
        ]), 2, 0)

    return {
        k: fracfun(corr['num'], corr['den'])
        for k, fracfun in fracfuns.iteritems()
    }


def process_1d(opts):
    print opts['key']

    defaults = {k: v['default'] for k, v in opts['params'].iteritems()}

    return opts['key'], [
        fraction_synchrony(dict(
            num=dict_merge(defaults, {opts['key']: val, 'Spp': 0, 'mp': 0}),
            den=dict_merge(defaults, {opts['key']: val})
        )) for val in opts['params'][opts['key']]['range']
    ]
    

def make_products_1d(params=None, pool=None, **_):
    return collections.OrderedDict(poolmap(pool, process_1d, [
        dict(key=key, params=params) for key in params.iterkeys()
    ]))


def plot_products_1d(params=None, products=None, filename=None):
    fig = pp.figure(figsize=(10, 15))
    fig.suptitle('Frac. avg. sync. only para moran/dispersal / all effects')

    n = len(params)

    for i, (k, u) in enumerate(products.iteritems()):
        host = [uu['fracavgsync'][0, 1] for uu in u]

        pp.subplot(n, 1, i+1)
        pp.xlabel('$%s$ (%s)' % (
            models.parasitism.symbols[k],
            models.parasitism.labels[k]
        ))
        pp.plot(params[k]['range'], host, label='host')
        pp.axvline(params[k]['default'], ls=':', color='r')
        pp.axhline(
            np.interp(params[k]['default'], params[k]['range'], host),
            ls=':', color='r'
        )
        pp.ylim(min(host), max(host))
        pp.xlim(min(params[k]['range']), max(params[k]['range']))

        if i == 0:
            pp.legend()

    pp.subplots_adjust(hspace=1.25)
    pp.savefig(filename)


def dict_merge(a, b):
    c = a.copy()
    c.update(b)
    return c


def process_2d(opts):
    defaults = {k: v['default'] for k, v in opts['params'].iteritems}
    range1 = opts['params'][opts['k1']]['range']
    range2 = opts['params'][opts['k2']]['range']
    k1, k2 = opts['k1'], opts['k2']
    res = opts['res']

    print k1, k2, res

    return [
        [
            fraction_synchrony(dict(
                num=dict_merge(defaults, {
                    k1: range1[i1], k2: range2[i2],
                    'Spp': 0, 'mp': 0
                }),
                den=dict_merge(defaults, {k1: range1[i1], k2: range2[i2]})
            ))
            for i2 in range(res)
        ] for i1 in range(res)
    ]


def make_products_2d(params=None, pool=None, res=40):
    keyproduct = list(itertools.combinations_with_replacement(params.keys(), 2))

    products = poolmap(pool, process_2d, [
        dict(params=params, k1=k1, k2=k2, res=res) for k1, k2 in keyproduct
    ])

    return [
        [
            [
                x for i, x in enumerate(products)
                if (keyproduct[i][0] == k1 and keyproduct[i][1] == k2)
                or (keyproduct[i][0] == k2 and keyproduct[i][1] == k1)
            ][0]
            for k2 in params.iterkeys()
        ] for k1 in params.iterkeys()
    ]


def plot_products_2d(params=None, products=None, filename=None):
    fig = pp.figure(figsize=(30, 30))
    fig.suptitle('Frac. avg. sync. only para moran/dispersal / all effects')
    n = len(params)

    for i in range(n):
        for j in range(i, n):
            ki, kj = params.keys()[i], params.keys()[j]
            pp.subplot(n, n, n * n - i * n - j)

            if i == 0:
                pp.xlabel('$%s$ (%s)' % (
                    models.parasitism.symbols[kj],
                    models.parasitism.labels[kj]
                ))

            if j == n - 1:
                pp.ylabel('$%s$ (%s)' % (
                    models.parasitism.symbols[ki],
                    models.parasitism.labels[ki]
                ))

            host = np.array([
                [u3['fracavgsync'][0, 1] for u3 in u2]
                for u2 in products[i][j]
            ])

            pp.imshow(
                host,
                label='host', cmap='rainbow', vmin=0, aspect='auto',
                extent=(
                    params[ki]['range'][0], params[ki]['range'][-1],
                    params[kj]['range'][0], params[kj]['range'][-1]
                )
            )

            pp.axvline(params[kj]['default'], color='pink', ls=':', lw=2)
            pp.axhline(params[ki]['default'], color='pink', ls=':', lw=2)

            pp.xlim(params[kj]['range'][0], params[kj]['range'][-1])
            pp.ylim(params[ki]['range'][0], params[ki]['range'][-1])

            pp.colorbar()

    pp.subplots_adjust(hspace=0.75)
    pp.savefig(filename)


def main():
    res = 100

    params = collections.OrderedDict(
        r=dict(default=2.0, range=(1.1, 4.0)),
        a=dict(default=1.0, range=(0.1, 2.0)),
        c=dict(default=1.0, range=(0.1, 2.0)),
        k=dict(default=0.5, range=(0.1, 0.9)),
        mh=dict(default=0.25, range=(0.125, 0.5)),
        mp=dict(default=0.25, range=(0.125, 0.5)),
        Sh=dict(default=0.5, range=(0.125, 1.0)),
        Shh=dict(default=0.25, range=(0.125, 1.0)),
        Sp=dict(default=0.5, range=(0.125, 1.0)),
        Spp=dict(default=0.25, range=(0.125, 1.0))
    )

    for p in params.itervalues():
        p['range'] = np.linspace(p['range'][0], p['range'][1], res)

    dims = int(os.environ.get('DIMS', 1))

    pool = multiprocessing.Pool(
        processes=os.environ.get('POOLCPUS', multiprocessing.cpu_count() - 1)
    )

    plotfun = plot_products_1d if dims is 1 else plot_products_2d
    makefun = make_products_1d if dims is 1 else make_products_2d

    cachepath = 'cache/fraction-host-%d.pickle' % dims
    plotpath = 'plots/fraction-host-%d.png' % dims

    for path in [cachepath, plotpath]:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    if not os.path.exists(cachepath):
        products = makefun(
            params=params,
            pool=pool,
            res=res
        )
    else:
        products = cPickle.load(open(cachepath))

    plotfun(
        params=params,
        products=products,
        filename=plotpath
    )

if __name__ == '__main__':
    main()
