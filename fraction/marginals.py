"""
marginals.py
parasynchrony
2015 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Usage: python marginals.py command configname.json [start] [stop]

    command={genruns, run, plot}
        genruns: generate a qsub script for all runs. The "slice size" (how many
            computations to perform per run) is defined in the config file.
        run: run the given configuration for a single slice, starting with an
            offset computation # start and ending with stop - 1.
        plot: plot the saved results.

    configname.json: configuration file describing model parameters and ranges,
        computation resolution, number of processes used, and plotting
        preferences.

Produces intermediate cached pickle files in the same path as the config
file, defaulting to cache/. See files in configs/fraction for some examples.
Leaving certain parameters out resorts to their defaults. Read
configs/fraction/schema.json for info on how these JSON files should be
formatted.

TODO: Time permitting, it would probably be best to use a single HDF5 file
    rather than summing up a bunch of incremental cached files.
TODO: May be smart to otherwise be using sparse arrays.
"""

import os
import sys
import time
import json
import pprint
import cPickle
import warnings
import itertools
import collections

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as pp

import models
import utilities

model = models.parasitism.get_model('nbd(2)')
model.lambdify_ss = False
printer = pprint.PrettyPrinter()

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

def plot_marginals(config):
    """
    TODO: This.

    :param config:
    """

    pass

def run_slice(config, start, stop):
    """
    Where the action is.

    :param config: (dict) configuration dictionary provided by openconfig.
    :param start: (int) start index in the list of combos of param values.
    :param stop: (int) stop index (uninclusive) in the list of combos of param
        values.
    """

    cacheprefix = os.path.join(config['file']['dir'], config['file']['name'])
    cachepath = '%s-%d-%d.pickle' % (cacheprefix, start, stop)

    params = config['params']
    paramkeys = config['props']['paramkeys']
    nparams = len(paramkeys)

    res, marginaldims, samplings = [
        config['args'][k] for k in ['res', 'marginaldims', 'samplings']
    ]

    # Parameters that actually vary, their count, and their index within the
    # ordered parameter list.
    varkeys = config['props']['varkeys']
    nvarkeys = len(varkeys)
    varkeyindices = {k: i for i, k in enumerate(varkeys)}

    # All possible combinations of varying parameters. Values are the index of
    # the value for the particular parameter. It's an iterator, so we can freely
    # slice it without having to actually store the thing in memory.
    pcombos = itertools.product(range(res), repeat=marginaldims)
    runslice = pcombos[start:stop]

    # TODO: This is messy. Ideally, I'd be able to change which metrics are
    # returned in compute_metrics and have the structure of these histograms
    # automatically change. Easy fix is to wait to create the matrices until we
    # see the first dict of values returned.
    popkeys = ['h', 'p']
    effectkeys = ['Rhh', 'Rpp']

    # Construct the statistic matrices.
    # varindex1, varindex2, ..., paramindex1, paramindex2, ...
    histshape = (nvarkeys,) * marginaldims + (res,) * marginaldims

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
                    histshape + (sampling['res'],), dtype=int
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
                    (int(len(runslice) * sampling['p']), nparams + 1),
                    dtype=float
                )

                # How many samples we have left to compute. Decrements. Note
                # that this may not actually reach zero, so it'll be important
                # to check it when plotting. Samples will start from the end,
                # as they are placed at samplesleft.
                samplesleft[popkey][effectkey][sampkey] = int(
                    len(runslice) * sampling['p']
                )

    for indices in runslice:
        # Make a single parameter set for this computation, setting each param
        # value to the value in its list of possible params at the index
        # specified.
        paramset = collections.OrderedDict(params)
        for ki, paramname in enumerate(params):
            paramset[paramname] = params[paramname][indices[ki]]

        metrics = compute_metrics(paramset)

        for popkey, effectkey, sampkey in itertools.product(
                popkeys, effectkeys, samplings
        ):
            ind_samplesleft = samplesleft[popkey][effectkey][sampkey]
            ind_samples = samples[popkey][effectkey][sampkey]
            ind_maxima = maxima[popkey][effectkey][sampkey]
            ind_counts = counts[popkey][effectkey][sampkey]
            metric = metrics[popkey][effectkey]

            (bmin, bmax), bres, incmin, recordp = [
                samplings[sampkey][k] for k in ['range', 'res', 'inclmin', 'p']
            ]

            # Bin the metric in a histogram for this sampling range. Really,
            # the bin index only needs to be computed once per sampling range,
            # irrespective of which marginal we are computing, but oh well.
            binprops = dict(left=np.nan, right=np.nan)

            if incmin:
                binindex = np.interp(metric, [bmin, bmax], [0, res], **binprops)
            else:
                binindex = 1 + np.interp(
                    metric, [bmin, bmax], [0, res - 1], **binprops
                )

            # Randomly store some fraction of values for each sampling range.
            if (samplesleft[popkey][effectkey][sampkey] > 0) and (
                np.random.rand(0, 100) <= (recordp * 100)
            ):
                ind_samples[ind_samplesleft] = paramset.values() + [metric]
                samplesleft[popkey][effectkey][sampkey] -= 1

            # Store statistics for each marginal parameter combination.
            for vkeys in itertools.product(varkeys, nvars):
                paramindices = tuple([
                    indices[varkeyindices[vk]] for vk in vkeys
                ])

                # Record the maximum value and its respective parameters for
                # each marginal.
                if metric > maxima[vkeys + paramindices]:
                    ind_maxima[
                        vkeys + paramindices
                    ] = [metric] + [
                        params[vk][paramindices[varkeyindices[vk]]]
                        for vk in varkeys
                    ]

                # Increment the counts for this marginal.
                ind_counts[vkeys + paramindices + (binindex,)] += 1

    cPickle.dump(
        dict(
            counts=counts,
            maxima=maxima,
            samples=samples,
            samplesleft=samplesleft
        ),
        open(cachepath, 'w')
    )

def open_config(configpath=None):
    """
    Open a configuration file and reformat it.

    :param configpath: path to configuration JSON file.
    :return: (dict) Configuration dictionary with parameter ranges replaced with
        lists of parameter values and a few extra keys:
        file:
            dir: directory to store cached files.
            name: filename prefix for cached files.
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
    else:
        config = dict(
            file=dict(
                dir='cache/',
                name='fracsync-marginals-default'
            ),
            args=dict(
                resolution=10,
                samplings=dict(
                    zero_one=dict(range=[0,1], res=100, inclmin=True, p=0),
                    one_ten=dict(
                        range=[1,10], res=100, inclmin=False, p=0.01
                    ),
                    gt_ten=dict(range=[10,], res=1, inclmin=False, p=0.01)
                ),
                marginaldims=2
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
        p.setdefault('res', res)
        p.setdefault('range', (p['default'], p['default']))

        if p['range'][0] != p['range'][1]:
            p['range'] = np.linspace(p['range'][0], p['range'][1], res)
        else:
            p['range'] = [p['default']]

    config['props'] = dict(paramkeys=config['params'].keys())
    config['props']['varkeys'] = [
        k for k in config['props']['paramkeys'] if len(config['params'][k]) > 1
    ]

    return config

def generate_runs(config):
    """
    TODO: Generate a qsub file that runs a bunch of slices.

    :param config:
    :return:
    """

    pass

def main():
    """Main."""

    try:
        config = open_config(sys.argv[2])

        if sys.argv[1] == 'run':
            start, stop = sys.argv[3:]
            run_slice(config, int(start), int(stop))
        elif sys.argv[1] == 'genruns':
            generate_runs(config)
    except IndexError:
        print 'usage: python marginals.py {genruns, runs, plot}', \
            'config.json [start] [stop]'

if __name__ == '__main__':
    main()
