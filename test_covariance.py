"""
test_covariance.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey
"""

import sys
import json
import os.path
import cPickle
import itertools
import multiprocessing

import numpy as np
import matplotlib
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as pp

import models


def cov_integrated(model, noise, params, maxfreqs=2**12):
    """
    Compute covariance numerically integrated from analytic spectrum with
    increasing frequency resolution. This should converge upon the analytic
    covariance.

    :param model: (models.stochastic.StochasticModel) the model.
    :param noise: (np.ndarray) noise covariance.
    :param params: (dict) parameter values, keyed by their sympy symbols.
    :param maxfreqs: (int) F, maximum number of frequencies to evaluate
        (default: 4096).
    :return (np.ndarray, np.ndarray): (F, N, N) array of covariance arrays (
        where N is the number of state variables) and a (F,) array of the number
        of frequencies over which each was integrated.
    """

    nfreqs = 2**np.arange(4, int(np.log2(maxfreqs)))

    cov = np.array([
        model.integrate_covariance_from_analytic_spectrum(
            params, noise, f
        ) for f in nfreqs
    ])

    return cov, nfreqs


def pool_cov(data):
    """
    Simulate the linear model and compute an estimate of its covariance. This
    is to be used with multiprocessing.Pool for parallel computation.

    :param data: (model, params, noise, ts) tuple where model is the
        models.StochasticModel instance, params is the dictionary of parameter
        values, keyed by their sympy symbols, noise is an np.ndarray of noise
        covariance, and ts is the number of timesteps to simulate.
    :return (np.ndarray): the covariance estimate for the simulated data.
    """

    model, params, noise, ts = data
    return np.cov(
        model.simulate_linear(np.zeros(len(model.vars)), params, noise, ts)
    )


def cov_simulated(model, noise, params, maxtimesteps=2**12, trials=2**6):
    """
    Simulate the linear model multiple times for an increasing number of
    timesteps to obtain a distribution of covariances for each number of
    timesteps. This should converge upon the analytic covariance.

    :param model: (models.stochastic.StochasticModel) the model
    :param noise: (np.ndarray) noise covariance.
    :param params: (dict) parameter values, keyed by their sympy symbols.
    :param maxtimesteps: (int) T, maximum number of timesteps to simulate
        (default: 4096).
    :param trials: (int) S, the number of simulation trials for each number of
        timesteps (default: 64).
    :return: (np.ndarray, np.ndarray) (T, S, N, N) array of covariances (where
        N is the number of state variables) and a (T,) array of numbers of
        timesteps for which each set of S covariaces was computed.
    """

    timesteps = 2**np.arange(4, int(np.log2(maxtimesteps)))

    nvars = len(model.vars)

    covariances = np.zeros((len(timesteps), trials, nvars, nvars))

    pool = multiprocessing.Pool()

    for i, ts in enumerate(timesteps):
        print 'Simulating %d timesteps.' % ts
        covariances[i] = pool.map(
            pool_cov,
            [(model, params, noise, ts)] * trials
        )

    return covariances, timesteps


def plot_convergences(
        model=None,
        analytic=None,
        integrated=None,
        simulated=None,
        nfreqs=None,
        nsteps=None,
        plotpath='covariance-test.pdf',
        **_
):
    """
    Plot convergence of a) covariance from numerical integration of the analytic
    cospectrum over an increasing number of frequencies and b) the distribution
    of covariance estimates of simulated data for an increasing number of
    timesteps upon the analytic covariance result.

    Saves a plot to plotpath.

    :param model: (models.stochastic.StochasticModel) the model.
    :param plotpath: (str) path at which to save the plot (default:
        covariance-test.pdf).
    :param analytic: (np.ndarray) (N, N) analytic covariance.
    :param integrated: (np.ndarray) (F, N, N) integrated covariances.
    :param simulated: (np.ndarray) (T, trials, N, N) simulated covariances.
    :param nfreqs: (np.ndarray) (F,) list of number frequencies integrated over.
    :param nsteps: (np.ndarray) (T,) list of number of timesteps simulated.
    """

    # 1. Plot.
    evensteps = [t for t in nsteps if int(np.log2(t)) % 2 == 0]
    evenfreqs = [f for f in nfreqs if int(np.log2(f)) % 2 == 0]

    nvars = len(model.vars)

    pp.figure(figsize=(15, 15))

    ax1, ax2 = None, None

    colors = dict(
        simulated=np.array((.344, .652, .418)),
        integrated=np.array((.306, .453, .680)),
        analytic=np.array((0, 0, 0))
    )

    for i, j in itertools.combinations_with_replacement(range(nvars), 2):
        ax1 = pp.subplot(nvars, nvars, j * nvars + i + 1)

        # a. Simulated.
        y1 = np.percentile(simulated[:, :, i, j], 1, axis=1)
        y99 = np.percentile(simulated[:, :, i, j], 99, axis=1)
        ymean = np.mean(simulated[:, :, i, j], axis=1)
        y25 = np.percentile(simulated[:, :, i, j], 25, axis=1)
        y75 = np.percentile(simulated[:, :, i, j], 75, axis=1)

        ax1.fill_between(nsteps, y1, y99, color=colors['simulated'], alpha=0.25)
        ax1.fill_between(
            nsteps, y25, y75, color=colors['simulated'], alpha=0.25
        )
        ax1.plot(nsteps, ymean, color=colors['simulated'], label='simulated')

        ax1.set_xlim(nsteps[0], nsteps[-1])
        ax1.set_xscale('log')
        ax1.set_xlabel(r'timesteps $(T)$', color=colors['simulated'])
        ax1.set_xticks(evensteps)
        ax1.set_xticklabels(['$2^{%d}$' % int(t) for t in np.log2(evensteps)])

        # b. Numerically integrated.
        ax2 = ax1.twiny()

        ax2.plot(
            nfreqs, integrated[:, i, j],
            color=colors['integrated'], label='integrated'
        )
        ax2.set_xlabel('frequencies $(F)$', color=colors['integrated'])
        ax2.set_xlim(nfreqs[0], nfreqs[-1])
        ax2.set_xscale('log')
        ax2.set_xticks(evenfreqs)
        ax2.set_xticklabels(['$2^{%d}$' % int(f) for f in np.log2(evenfreqs)])

        # c. Analytic.
        ax2.axhline(analytic[i, j], color=colors['analytic'], label='analytic')
        ax1.set_ylabel('$\\Sigma_{%s%s}$' % (model.vars[i], model.vars[j]))

    # 2. Legend.
    pp.subplot(nvars, nvars, nvars)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    pp.legend(handles1 + handles2, labels1 + labels2)
    pp.axis('off')

    # 3. Adjust and save plot.
    pp.subplots_adjust(wspace=0.5, hspace=0.5)

    print 'Saving %s.' % plotpath
    pp.savefig(plotpath)


def main():
    """usage: python covariance_test.py {config.json, output.pickle}"""

    try:
        extension = os.path.splitext(sys.argv[1])[1]
        plotargs = dict(maxfreqs=2**16, maxtimesteps=2**16, trials=2**8)

        # 1. Either compute or load precomputed data.
        if extension == '.json':
            config = json.load(open(sys.argv[1]))
            sim = config['simulation']

            # a. Create the model from the loaded configuration.
            plotargs['model'] = models.parasitism.get_model(sim['model'])
            plotargs['params'] = models.parasitism.sym_params(sim['params'])
            plotargs['noise'] = np.array([
                [sim['noise']['Sh'], sim['noise']['Shh'], 0, 0],
                [sim['noise']['Shh'], sim['noise']['Sh'], 0, 0],
                [0, 0, sim['noise']['Sp'], sim['noise']['Spp']],
                [0, 0, sim['noise']['Spp'], sim['noise']['Sp']]
            ])

            # b. Compute everything.
            plotargs['analytic'] = plotargs['model'].calculate_covariance(
                plotargs['params'], plotargs['noise']
            )

            plotargs['integrated'], plotargs['nfreqs'] = cov_integrated(
                plotargs['model'],
                plotargs['noise'],
                plotargs['params'],
                plotargs['maxfreqs']
            )

            plotargs['simulated'], plotargs['nsteps'] = cov_simulated(
                plotargs['model'],
                plotargs['noise'],
                plotargs['params'],
                plotargs['maxtimesteps'],
                plotargs['trials']
            )
        elif extension == '.pickle':
            # c. If pickle, load the saved data before plotting.
            plotargs = cPickle.load(open(sys.argv[1]))

        pathbase = os.path.join(
            os.path.dirname(sys.argv[1]),
            'covariance-test-maxfreqs-%d-maxtimesteps-%d-trials-%d' % (
                plotargs['maxfreqs'],
                plotargs['maxtimesteps'],
                plotargs['trials']
            )
        )

        # 2. Make the plot.
        plotargs['plotpath'] = pathbase + '.pdf'
        plot_convergences(**plotargs)

        # 3. Save pickled data if it was computed.
        if extension != 'pickle':
            picklepath = pathbase + '.pickle'

            print 'Writing %s.' % picklepath
            cPickle.dump(plotargs, open(picklepath, 'w'))
    except IndexError:
        print "Must have either .json or .pickle file as argument."

if __name__ == '__main__':
    main()