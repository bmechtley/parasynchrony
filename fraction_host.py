import itertools
import multiprocessing

import numpy as np
import matplotlib.pyplot as pp

import models

model = models.parasitism.get_model('nbd(2)')


def fraction_synchrony(params, noise, nfreqs=100):
    """
    Compute the fraction of synchrony between patches for which the host is
    responsible using different metrics.

    TODO: Allow caching of spectra.
    TODO: Return fraction for which different populations are responsible.
    TODO: Optimize.
    TODO: Allow for different parameter ranges.

    :param params: model parameter dictionary. Contains "num" and "den" keys
        that contain parameter keys.
    :param noise: noise parameters (H/P variance, H-H/P-P covariance)
    :param nfreqs: number of frequency values to compute.
    :return: dictionary of metric: fraction pairs.
    """

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

        noisecov = np.array([
            [noise[name]['Sh'], noise[name]['Shh'], 0, 0],
            [noise[name]['Shh'], noise[name]['Sh'], 0, 0],
            [0, 0, noise[name]['Sp'], noise[name]['Spp']],
            [0, 0, noise[name]['Spp'], noise[name]['Sp']]
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

    params = dict(
        r=np.linspace(0.1, 1.0, res),
        a=np.linspace(0.1, 10, res),
        c=np.linspace(0.1, 10, res),
        k=np.linspace(0.1, 10, res),
        mh=np.linspace(0.1, 0.5, res),
        mp=np.linspace(0.1, 0.5, res),
        Sh=np.linspace(0, 1.0, res),
        Shh=np.linspace(0, 1.0, res),
        Sp=np.linspace(0, 1.0, res),
        Spp=np.linspace(0, 1.0, res)
    )

    product = itertools.product(*params.values())
    nslice = 8

    pool = multiprocessing.Pool(processes=4)

    result = []

    while True:
        resslice = pool.map(
            fraction_synchrony, itertools.islice(product, nslice)
        )
    combos = [
        pair
        for pair in itertools.product(*params.values())
    ]

    print len(combos), combos[0], combos[1]