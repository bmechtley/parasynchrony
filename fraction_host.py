import os
import cPickle
import itertools
import multiprocessing

import numpy as np
import matplotlib.pyplot as pp

import models

model = models.parasitism.get_model('nbd(2)')


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
        fracavgsync=lambda n, d: np.mean(n, axis=0) / np.mean(d, axis=0)
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


def make_products_1d(params, defaults):
    products = []

    for key in defaults.iterkeys():
        pr = []

        for val in params[key]:
            dd = dict(defaults)
            dd[key] = val
            param_set = dict(num=dict(dd), den=dict(dd))
            param_set['num'].update(dict(Spp=0, mp=0))

            pr.append(fraction_synchrony(param_set))

        products.append(pr)

    return products


def plot_products_1d(keys, labels, params, products, filename):
    pp.figure(figsize=(10, 15))

    for i, (k, u) in enumerate(zip(keys, products)):
        pp.subplot(len(keys), 1, i+1)
        pp.xlabel(labels[i])
        host = [uu['fracavgsync'][0, 1] for uu in u]
        para = [uu['fracavgsync'][2, 3] for uu in u]

        pp.plot(params[k], host, label='host')
        pp.plot(params[k], para, label='para')
        pp.ylim(0, max(max(host), max(para)) + 1)

        if i == 0:
            pp.legend()
            pp.title('Frac. avg. sync. only para moran/dispersal / all effects')

    pp.subplots_adjust(hspace=0.75)
    pp.savefig(filename)


def helper_2d(defaults=None, params=None, k1=None, k2=None, i1=None, i2=None):
    dd = dict(defaults)
    dd[k1] = params[k1][i1]
    dd[k2] = params[k2][i2]
    param_set = dict(num=dict(dd), den=dict(dd))
    param_set['num'].update(dict(Spp=0, mp=0))

    return fraction_synchrony(param_set)


def dict_merge(a, b):
    c = a.copy()
    c.update(b)
    return c


def mapped(opts):
    print opts['k1'], opts['k2']

    return [
        [
            fraction_synchrony(dict(
                num=dict_merge(opts['defaults'], {
                    opts['k1']: opts['params'][opts['k1']][i1],
                    opts['k2']: opts['params'][opts['k2']][i2],
                    'Spp': 0, 'mp': 0
                }),
                den=dict_merge(opts['defaults'], {
                    opts['k1']: opts['params'][opts['k1']][i1],
                    opts['k2']: opts['params'][opts['k2']][i2]
                })
            ))
            for i2 in range(opts['res'])
        ] for i1 in range(opts['res'])
    ]


def make_products_2d(keys, params, defaults, res=40):
    keyproduct = itertools.combinations_with_replacement(keys, 2)

    pool = multiprocessing.Pool(processes=os.environ.get(
        'POOLPROCESSES', multiprocessing.cpu_count() - 1
    ))

    products = pool.map(mapped, [
        dict(
            params=params,
            defaults=defaults,
            k1=k1,
            k2=k2,
            res=res
        ) for k1, k2 in keyproduct
    ])

    products = [
        [
            next(
                x for i, x in enumerate(products)
                if keyproduct[i][0] == k1
                and keyproduct[i][1] == k2
            )
            for k2 in keys
        ] for k1 in keys
    ]

    return products


def plot_products_2d(keys, labels, params, products, filename):
    pp.figure(figsize=(10, 15))

    lk = len(keys)
    for i, (i1, k1), (i2, k2) in enumerate(
            itertools.combinations_with_replacement(enumerate(keys), 2)
    ):
        pp.subplot(lk, lk, i1 * lk + i2 + 1)

        if i1 == lk - 1:
            pp.xlabel(labels[i2])

        if i2 == 0:
            pp.ylabel(labels[i1])

        host = [
            [uuu['fracavgsync'][0, 1] for uuu in uu] for uu in products[i1][i2]
        ]
        # para = [[uuu['fracavgsync'][2, 3] for uuu in uu] for uu in u]

        pp.imshow(params[k1], params[k2], host, label='host')
        # pp.imshow(params[k1], params[k2], para, label='para')

        # pp.ylim(0, max(max(host), max(para)) + 1)

        if i == 0:
            pp.title('Frac. avg. sync. only para moran/dispersal / all effects')

    pp.subplots_adjust(hspace=0.75)
    pp.savefig(filename)


def main():
    res = 50

    defaults = dict(
        r=3.0,
        a=0.5,
        c=1.2,
        k=0.9,
        mh=0.05,
        mp=0.05,
        Sh=1E-2,
        Shh=1E-4,
        Sp=1E-2,
        Spp=1E-4
    )

    params = dict(
        r=np.linspace(1.1, 10, res),  # r
        a=np.linspace(0.1, 10, res),  # a
        c=np.linspace(0.1, 10, res),  # c
        k=np.linspace(0.1, 0.9, res),  # k - stable iff < 1
        mh=np.linspace(0, 0.5, res),  # mh
        mp=np.linspace(0, 0.5, res),  # mp
        Sh=np.linspace(0, 1.0, res),  # Sh
        Shh=np.linspace(0, 1.0, res),  # Shh
        Sp=np.linspace(0, 1.0, res),  # Sp
        Spp=np.linspace(0, 1.0, res)  # Spp
    )

    keys = ['r', 'a', 'c', 'k', 'mh', 'mp', 'Sh', 'Sp', 'Shh', 'Spp']
    labels = [
        '$\lambda$ (host reproduction rate)',
        'a (para attack range)',
        'c (# eggs per parasitized host)',
        'k (host clumping)',
        r'$\mu_H$ (host migration)',
        r'$\mu_P$ (para migration)',
        r'$\Sigma_{H}$ (host intra-patch env. variance)',
        r'$\Sigma_{P}$ (para intra-patch env. variance)',
        r'$\Sigma_{HH}$ (host inter-patch env. covariance)',
        r'$\Sigma_{PP}$ (para inter-patch env. covariance)'
    ]

    if not os.path.exists('results.pickle'):
        products = make_products_2d(keys, params, defaults, res)
        cPickle.dump(products, open('result.pickle', 'w'))
    else:
        products = cPickle.load(open('result.pickle'))

    plot_products_2d(
        keys, labels, params, products, 'fracsync-host.pdf'
    )

if __name__ == '__main__':
    main()
