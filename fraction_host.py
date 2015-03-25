import os
import cPickle
import itertools
import multiprocessing

import numpy as np

import models

model = models.parasitism.get_model('nbd(2)')


def fraction_synchrony(p, nfreqs=100):
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

    params = dict(zip(
        ['r', 'a', 'c', 'k', 'mh', 'mp', 'Sh', 'Shh', 'Sp', 'Spp'],
        p[:10]
    ))

    print params

    params = dict(num=dict(params), den=dict(params))

    params['den'].update(dict(
        mh=0.5, mp=0.5,
        Sh=1.0, Shh=0.5,
        Sp=1.0, Spp=0.5
    ))

    fracfuns = dict(
        avgfracsync=lambda(n, d): np.mean(n / d, axis=0),
        maxfracsync=lambda(n, d): np.amax(n / d, axis=0),
        fracavgsync=lambda(n, d): np.mean(n, axis=0) / np.mean(d, axis=0),
        fracmaxsync=lambda(n, d): np.amax(n, axis=0) / np.amax(d, axis=0)
    )

    freqs = np.linspace(0, 0.5, nfreqs)
    corr = dict()

    for name in ['num', 'den']:
        sym_params = models.parasitism.sym_params(params[name])

        noisecov = np.array([i
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


if __name__ == '__main__':
    res = 10

    params = [
        np.linspace(0.1, 1.0, res),  # r
        np.linspace(0, 10, res),  # a
        np.linspace(0, 10, res),  # c
        np.linspace(0, 10, res),  # k
        np.linspace(0, 0.5, res),  # mh
        np.linspace(0, 0.5, res),  # mp
        np.linspace(0, 1.0, res),  # Sh
        np.linspace(0, 0.5, res),  # Shh
        np.linspace(0, 1.0, res),  # Sp
        np.linspace(0, 0.5, res)  # Spp
    ]

    print params

    pool = multiprocessing.Pool(processes=os.environ.get(
        'POOLPROCESSES', multiprocessing.cpu_count() - 1
    ))

    print pool

    result = pool.map(fraction_synchrony, itertools.product(*params))

    cPickle.dump(result, open('result.pickle', 'w'))