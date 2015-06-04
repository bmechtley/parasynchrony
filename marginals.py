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
import glob
import pprint
import cPickle
import operator
import functools
import itertools

import numpy as np

import models
import utilities

model = models.parasitism.get_model('nbd(2)')
model.lambdify_ss = False
printer = pprint.PrettyPrinter()

def ncalcs(config):
    return functools.reduce(
        operator.mul,
        [len(param['range']) for param in config['params'].values()],
        1
    )

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
        pnum['m%s' % effects] = 0
        nnum['C%s%s' % ((effects,) * 2)] = 0

        cnum, cden = tuple([
            models.utilities.correlation(model.calculate_covariance(
                models.parasitism.sym_params(p), noise_cov(n)
            ))
            for p, n in [(pnum, nnum), (pden, nden)]
        ])

        cfrac = abs(cnum) / abs(cden)
        metrics[effects] = dict(Rhh=cfrac[0, 1], Rpp=cfrac[2, 3])

    return metrics

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

    paramres, samplings = [config['args'][k] for k in 'resolution', 'samplings']

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
    histshape = (nvarkeys, nvarkeys, paramres, paramres)

    counts, maxima, samples, samplesleft = [
        {
            sampkey: {
                popkey: {
                    effectkey: None
                    for effectkey in effectkeys
                } for popkey in popkeys
            } for sampkey in samplings
        }
        for _ in range(4)
    ]

    for sampkey, sampling in samplings.iteritems():
        nsamples = int(sampling['nsamples'])
        sampres = sampling['resolution']

        for popkey in popkeys:
            for effectkey in effectkeys:
                counts[sampkey][popkey][effectkey] = np.zeros(
                    histshape + (sampres,), dtype=int
                )

                # Store the list of argmax parameter values + the maximum metric
                # value.
                maxima[sampkey][popkey][effectkey] = np.zeros(
                    histshape + (nparams + 1,), dtype=float
                )

                # Random samples are for the entire hypercube and not for each
                # marginal. Store the list of parameter values + the metric
                # value.
                samples[sampkey][popkey][effectkey] = np.zeros(
                    (nsamples, nparams + 1), dtype=float
                )

                # How many samples we have left to compute. Decrements. Note
                # that this may not actually reach zero, so it'll be important
                # to check it when plotting. Samples will start from the end,
                # as they are placed at samplesleft.
                samplesleft[sampkey][popkey][effectkey] = nsamples

    return dict(
        counts=counts,
        maxima=maxima,
        samples=samples,
        samplesleft=samplesleft
    )

def param_product(config):
    """
    Return an iterator for a product of all parameter values given a config
    dictionary.

    :param config: (dict) configuration dictionary provided by
        utilities.config_defaults.
    :return: (iterable) iterator of all combinations of parameter values.
    """

    return itertools.product(
        range(config['args']['resolution']),
        repeat=len(config['props']['varkeys'])
    )

def run_slice(config, start, stop):
    """
    Where the action is.

    :param config: (dict) configuration dictionary provided by
        utilities.config_defaults.
    :param start: (int) start index in the list of combos of param values.
    :param stop: (int) stop index (uninclusive) in the list of combos of param
        values.
    """

    print start
    print stop

    bt = time.clock()

    params = config['params']
    paramkeys = config['props']['paramkeys']

    # Parameters that actually vary, their count, and their index within the
    # ordered parameter list.
    varkeys = config['props']['varkeys']
    varkeyindices = {k: i for i, k in enumerate(varkeys)}

    # All possible combinations of varying parameters. Values are the index of
    # the value for the particular parameter. It's an iterator, so we can freely
    # slice it without having to actually store the thing in memory.
    pcombos = param_product(config)
    runslice = itertools.islice(pcombos, start, stop)

    storage_arrays = zero_storage_arrays(config)
    counts, maxima, samples, samplesleft = [storage_arrays[k] for k in (
        'counts', 'maxima', 'samples', 'samplesleft'
    )]
    res, samplings = [config['args'][k] for k in 'resolution', 'samplings']
    popkeys, effectkeys = ('h', 'p'), ('Rhh', 'Rpp')

    for indices in runslice:
        # Make a single parameter set for this computation, setting each param
        # value to the value in its list of possible params at the index
        # specified.
        paramset = dict()
        for param in params:
            if len(params[param]['range']) == 1:
                paramset[param] = params[param]['range'][0]

        for ki, param in enumerate(varkeys):
            paramset[param] = params[param]['range'][indices[ki]]

        metrics = compute_metrics(paramset)

        for sampkey, popkey, effectkey in itertools.product(
                samplings, popkeys, effectkeys
        ):
            ind_samplesleft = samplesleft[sampkey][popkey][effectkey]
            ind_samples = samples[sampkey][popkey][effectkey]
            ind_maxima = maxima[sampkey][popkey][effectkey]
            ind_counts = counts[sampkey][popkey][effectkey]
            metric = metrics[popkey][effectkey]

            brange, bres, incmin, nsamples = [samplings[sampkey][k] for k in (
                'range', 'resolution', 'inclmin', 'nsamples'
            )]
            recordp = float(nsamples) / (stop - start)

            bmin, bmax = brange[0], brange[-1]

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
            if (samplesleft[sampkey][popkey][effectkey] > 0) and (
                np.random.rand(0, 100) <= (recordp * 100)
            ):
                ind_samples[ind_samplesleft] = paramset.values() + [metric]
                samplesleft[sampkey][popkey][effectkey] -= 1

            # Store statistics for each marginal parameter combination.
            for vkeys in itertools.product(varkeys, repeat=2):
                # indices of varkeys used for marginals.
                vkis = tuple([varkeyindices[vk] for vk in vkeys])
                vkargis = tuple([indices[vki] for vki in vkis])

                # Record the maximum value and its respective parameters for
                # each marginal.
                if metric > ind_maxima[vkis + vkargis + (-1,)]:
                    ind_maxima[vkis + vkargis] = [
                        paramset[pk] for pk in paramkeys
                    ] + [metric]

                # Increment the counts for this marginal.
                if np.isfinite(binindex):
                    ind_counts[vkis + vkargis + (binindex,)] += 1

    for sampkey in samplings:
        cPickle.dump(
            dict(
                counts=counts[sampkey],
                maxima=maxima[sampkey],
                samples=samples[sampkey],
                samplesleft=samplesleft[sampkey]
            ),
            open(
                '%s-%s-%d-%d.pickle' % (
                    os.path.join(
                        config['file']['dir'],
                        config['file']['name']
                    ),
                    sampkey,
                    start,
                    stop
                ),
                'w'
            )
        )

    print time.clock() - bt

def generate_runs(config, runtype='qsub'):
    """
    Generate a qsub file that runs a bunch of slices. Saves a shell script in
    the parent directory of the configuration file.

    :param config: (dict) configuration dict opened with config_defaults.
    :param type: (str) 'qsub' for a qsub configuration file or 'sh' for a simple
        shell script that launches a bunch of processes.
    """

    config_dir, config_name, slice_size = [
        config['file'][k] for k in 'dir', 'name', 'slice_size'
    ]
    config_prefix = os.path.join(config_dir, config_name)

    nc = ncalcs(config)

    if runtype is 'sh':
        script_path = config_prefix + '-runs.sh'
        print 'Writing %s.' % script_path

        outfile = open(script_path, 'w')
        outfile.writelines(['set -e\n'] + [
            'python marginals.py run %s %d %d\n' % (
                config_prefix + '.json', start, start + slice_size
            )
            for start in range(0, nc, slice_size)
        ])
        outfile.close()
    elif runtype is 'qsub':
        job_path = config_prefix + '-runs.sh'

        outfile = open(job_path, 'w')
        outfile.writelines([
            '#PBS -N %s\n' % config['file']['name'],
            '#PBS -l nodes=1,mem=1000m,walltime=1:00:00\n',
            '#PBS -m n\n',
            '#PBS -S /bin/bash\n',
            '#PBS -d %s\n' % os.getcwd(),
            '#PBS -e /users/mechtley/logs/%s.err\n' % config['file']['name'],
            '#PBS -o /users/mechtley/logs/%s.out\n' % config['file']['name'],
            '#PBS -t 0-%d\n' % ((nc + 1) / slice_size),
            ' '.join([
                'python -W ignore marginals.py run',
                os.path.join(
                    os.getcwd(),
                    config['file']['dir'],
                    config['file']['name'] + '.json'
                ),
                '$((PBS_ARRAYID * %d))' % slice_size,
                '$((PBS_ARRAYID * %d + %d))\n' % (slice_size, slice_size)
            ])
        ])

        outfile.close()

def gather_runs(config):
    """
    :param config:
    """

    print 'Collecting data from saved runs.'

    cacheprefix = os.path.join(config['file']['dir'], config['file']['name'])

    popkeys, effectkeys = ('h', 'p'), ('Rhh', 'Rpp')
    samplings = config['args']['samplings']
    storage_arrays = utilities.zero_storage_arrays(config)
    counts, maxima, samples, samplesleft = [storage_arrays[k] for k in (
        'counts', 'maxima', 'samples', 'samplesleft'
    )]

    # Gather statistic arrays in each run's cache file.
    for sampkey, sampling in samplings.iteritems():
        cfns = glob.glob(cacheprefix + '-%s-*-*.pickle' % sampkey)

        for i, cfn in enumerate(cfns):
            cf = cPickle.load(open(cfn))
            print '\t%d / %d: %s' % (i, len(cfns), cfn)

            for popkey in popkeys:
                for effectkey in effectkeys:
                    # Shorthand for global arrays over all cached values.
                    gsampsleft = samplesleft[sampkey][popkey][effectkey]
                    gsamps = samples[sampkey][popkey][effectkey]
                    gcounts = counts[sampkey][popkey][effectkey]
                    gmaxima = maxima[sampkey][popkey][effectkey]

                    # Shorthand for arrays local to this set of cached values.
                    csampsleft = cf['samplesleft'][popkey][effectkey]
                    csamps = cf['samples'][popkey][effectkey]
                    ccounts = cf['counts'][popkey][effectkey]

                    cmaxima = cf['maxima'][popkey][effectkey]
                    ncsamps = len(csamps) - csampsleft

                    # Increment histograms.
                    gcounts += ccounts

                    # Gather samples.
                    if ncsamps:
                        gsamps[gsampsleft-ncsamps:gsampsleft] = csamps
                        samplesleft[sampkey][popkey][effectkey] -= ncsamps

                    # Gather maxima.
                    joined = np.array([gmaxima, cmaxima])

                    argmaxima = np.tile(
                        np.argmax(joined[..., -1], axis=0)[..., np.newaxis],
                        (1,) * (len(gmaxima.shape) - 1) + (gmaxima.shape[-1],)
                    )

                    maxima[sampkey][popkey][effectkey] = np.where(
                        argmaxima, gmaxima, cmaxima
                    )

    cachepath = '%s-full.pickle' % cacheprefix

    cPickle.dump(
        dict(counts=counts, maxima=maxima, samples=samples),
        open(cachepath, 'w')
    )

def main():
    """Main."""

    config = utilities.config_defaults(sys.argv[2])

    if sys.argv[1] == 'run' and len(sys.argv) == 5:
        start, stop = sys.argv[3:]
        run_slice(config, int(start), int(stop))
    elif sys.argv[1] == 'genruns' and len(sys.argv) == 3:
        generate_runs(config, runtype='qsub')
    elif sys.argv[1] == 'gather' and len(sys.argv) == 3:
        gather_runs(config)
    else:
        print 'usage: python marginals.py {genruns, runs, plot}', \
            'config.json [start] [stop]'

if __name__ == '__main__':
    main()
