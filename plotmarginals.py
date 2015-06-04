import os
import sys
import json
import glob
import cPickle
import operator
import functools
import itertools
import collections

import numpy as np

import matplotlib.pyplot as pp
import matplotlib.cm

def open_config(configpath=None):
    """
    Open a configuration file and reformat it.

    :param configpath: path to configuration JSON file.
    :return: (dict) Configuration dictionary with parameter ranges replaced with
        lists of parameter values and a few extra keys:
        file:
            dir: directory to store cached files.
            name: filename prefix for cached files.
            slice_size: number of computations per individual run.
        props:
            paramkeys: ordered list of parameter names.
            varkeys: ordered list of parameter names for those that vary.
    """

    if configpath is not None:
        configdir, configfile = os.path.split(configpath)
        configname = os.path.splitext(configfile)[0]
        config = json.load(open(configpath))
        config.setdefault('file', dict())
        config['file'].setdefault('dir', configdir)
        config['file'].setdefault('name', configname)
        config['file'].setdefault('slice_size', 20)
    else:
        config = dict(
            file=dict(
                dir='cache/',
                name='fracsync-marginals-default',
                slice_size=20
            ),
            args=dict(
                resolution=10,
                samplings=dict(
                    zero_one=dict(range=[0,1], res=100, inclmin=True, p=0),
                    one_ten=dict(
                        range=[1,10], res=100, inclmin=False, p=0.01
                    ),
                    gt_ten=dict(range=[10,], res=1, inclmin=False, p=0.01)
                )
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

    # Make sure the params are ordered consistently so we can easily slice
    # combinations without having to actually store info regarding to which
    # parameters each computation should use.
    config['params'] = collections.OrderedDict(config['params'])
    res = config['args']['resolution']

    config['props'] = dict(varkeys=[])

    for p in config['params'].itervalues():
        p.setdefault('resolution', res)
        p.setdefault('range', (p['default'], p['default']))

        if p['range'][0] != p['range'][1]:
            p['range'] = np.linspace(p['range'][0], p['range'][1], res)
        else:
            p['range'] = [p['default']]

    config['props'] = dict(paramkeys=config['params'].keys())
    config['props']['varkeys'] = [
        k for k in config['props']['paramkeys'] if len(config['params'][k]['range']) > 1
    ]

    return config

def zero_storage_arrays(config):
    """

    :param config:
    :return:
    """

    paramkeys = config['props']['paramkeys']
    nparams = len(paramkeys)

    nruns = functools.reduce(
        operator.mul,
        [len(param) for param in config['params'].values()],
        1
    )

    res, samplings = [config['args'][k] for k in 'resolution', 'samplings']

    # TODO: This is messy. Ideally, I'd be able to change which metrics are
    # returned in compute_metrics and have the structure of these histograms
    # automatically change. Easy fix is to wait to create the matrices until we
    # see the first dict of values returned.
    popkeys, effectkeys = ('h', 'p'), ('Rhh', 'Rpp')

    # Parameters that actually vary, their count, and their index within the
    # ordered parameter list.
    varkeys = config['props']['varkeys']
    nvarkeys = len(varkeys)

    # Construct the statistic matrices.
    # varindex1, varindex2, paramindex1, paramindex2, ...
    histshape = (nvarkeys, nvarkeys, res, res)

    counts, maxima, samples, samplesleft = [
        {
            popkey: {
                effectkey: {
                    sampkey: None
                    for sampkey in samplings
                } for effectkey in effectkeys
            } for popkey in popkeys
        }
        for _ in range(4)
    ]

    for popkey in popkeys:
        for effectkey in effectkeys:
            for sampkey, sampling in samplings.iteritems():
                counts[popkey][effectkey][sampkey] = np.zeros(
                    histshape + (sampling['resolution'],), dtype=int
                )

                # Store the list of argmax parameter values + the maximum metric
                # value.
                maxima[popkey][effectkey][sampkey] = np.zeros(
                    histshape + (nparams + 1,), dtype=float
                )

                # Random samples are for the entire hypercube and not for each
                # marginal. Store the list of parameter values + the metric
                # value.
                samples[popkey][effectkey][sampkey] = np.zeros(
                    (int(nruns * sampling['p']), nparams + 1),
                    dtype=float
                )

                # How many samples we have left to compute. Decrements. Note
                # that this may not actually reach zero, so it'll be important
                # to check it when plotting. Samples will start from the end,
                # as they are placed at samplesleft.
                samplesleft[popkey][effectkey][sampkey] = int(
                    nruns * sampling['p']
                )

    return dict(
        counts=counts,
        maxima=maxima,
        samples=samples,
        samplesleft=samplesleft
    )

def sum_products(config):
    """

    :param config:
    """

    print 'Collecting data from saved runs.'

    cacheprefix = os.path.join(config['file']['dir'], config['file']['name'])

    popkeys, effectkeys = ('h', 'p'), ('Rhh', 'Rpp')
    samplings = config['args']['samplings']
    storage_arrays = zero_storage_arrays(config)
    counts, maxima, samples, samplesleft = [storage_arrays[k] for k in (
        'counts', 'maxima', 'samples', 'samplesleft'
    )]

    # Gather statistic arrays in each run's cache file.
    for cfn in glob.glob(cacheprefix + '-*-*.pickle'):
        cf = cPickle.load(open(cfn))
        for popkey in popkeys:
            for effectkey in effectkeys:
                for sampkey, sampling in samplings.iteritems():
                    # Shorthand for global arrays over all cached values.
                    gsampsleft = samplesleft[popkey][effectkey][sampkey]
                    gsamps = samples[popkey][effectkey][sampkey]
                    gcounts = counts[popkey][effectkey][sampkey]
                    gmaxima = maxima[popkey][effectkey][sampkey]

                    # Shorthand for arrays local to this set of cached values.
                    csampsleft = cf['samplesleft'][popkey][effectkey][sampkey]
                    csamps = cf['samples'][popkey][effectkey][sampkey]
                    ccounts = cf['counts'][popkey][effectkey][sampkey]

                    cmaxima = cf['maxima'][popkey][effectkey][sampkey]
                    ncsamps = len(csamps) - csampsleft

                    # Increment histograms.
                    try:
                        gcounts += ccounts
                    except ValueError:
                        print cfn
                        print gcounts.shape, ccounts.shape
                        exit(-1)

                    # Gather samples.
                    print dict(
                        gsampsleft=gsampsleft,
                        ncsamps=ncsamps,
                        csampsleft=csampsleft,
                        csamps=csamps.shape,
                        gsamps=gsamps.shape
                    )

                    if ncsamps:
                        gsamps[gsampsleft-ncsamps:gsampsleft] = csamps
                        samplesleft[popkey][effectkey][sampkey] -= ncsamps

                    # Gather maxima.
                    print dict(gmaxima=gmaxima.shape, cmaxima=cmaxima.shape)
                    joined = np.array([gmaxima, cmaxima])

                    try:
                        argmaxima = np.tile(
                            np.argmax(joined[..., -1], axis=0)[..., np.newaxis],
                            (1,) * (len(gmaxima.shape) - 1) + (gmaxima.shape[-1],)
                        )

                        print dict(joined=joined.shape, argmaxima=argmaxima.shape)
                    except ValueError:
                        print cfn
                        print id(maxima[popkey][effectkey][sampkey])
                        print joined[..., -1]
                        exit(-1)

                    maxima[popkey][effectkey][sampkey] = np.where(
                        argmaxima, gmaxima, cmaxima
                    )

    cachepath = '%s-full.pickle' % cacheprefix

    cPickle.dump(
        dict(counts=counts, maxima=maxima, samples=samples),
        open(cachepath, 'w')
    )

def plot_marginals(config):
    """
    TODO: This.

    :param config:
    """

    cacheprefix = os.path.join(config['file']['dir'], config['file']['name'])
    cachefile = '%s-full.pickle' % cacheprefix

    if not os.path.exists(cachefile):
        sum_products(config)

    print 'Plotting marginals.'

    gathered = cPickle.load(open('%s-full.pickle' % cacheprefix))
    hists = gathered['counts']['h']['Rhh']

    varkeys, paramkeys = [config['props'][k] for k in 'varkeys', 'paramkeys']
    pp.figure(len(varkeys) * 15, len(varkeys) * 10)

    for spi, ((vki1, vk1), (vki2, vk2)) in enumerate(
        itertools.combinations_with_replacement(enumerate(varkeys), repeat=2)
    ):
        ax = pp.add_subplot(
            len(varkeys),
            len(varkeys),
            vki1 * len(varkeys) + vki2,
            projection='3d'
        )

        pp.ylabel(vk1)
        pp.xlabel(vk2)
        pp.ylim(np.amin(config['params'][vk1]), np.amax(config['params'][vk1]))
        pp.xlim(np.amin(config['params'][vk2]), np.amax(config['params'][vk2]))
        mx, my = np.meshgrid(config['params'][vk1], config['params'][vk2])

        hist = hists['zero_one'][vki1, vki2]     # res x res x nbins
        cumsums = np.cumsum(hist, axis=2)

        percs = 1, 5, 25, 50, 75, 95, 99
        colors = [matplotlib.cm.spectral(p) for p in percs]
        sampling = config['args']['samplings']['zero_one']

        for perc, color in zip(percs, colors):
            bin_idx = np.searchsorted(cumsums, np.percentile(cumsums, perc))
            vals = np.interp(
                bin_idx, [0, sampling['resolution'] - 1], sampling['range']
            )

            ax.plot_surface(mx, my, vals, color=color, alpha=0.5)

    pp.savefig('%s-zero-one.png' % cacheprefix)

def main():
    """Main."""

    plot_marginals(open_config(sys.argv[1]))

if __name__ == '__main__':
    main()
