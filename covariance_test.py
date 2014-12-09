import sys
import itertools
import json
import multiprocessing

import numpy as np
import matplotlib.pyplot as pp

import models


def plot_numerically_integrated(model, noise, params, maxfreqs=2**12):
    """
    Plot convergence test for numerically integrated cospectrum. Should line up
    with the analytic covariance. Saves a plot called
    plots/covariance-test-integration.pdf.

    :param model (models.StochasticModel): model
    :param noise (np.ndarray): noise covariance.
    :param params (dict): parameters for the model, keyed by SymPy symbols.
    :param maxfreqs (int): maximum number of frequency values to integrate.
    """

    nfreqs_trials = 2**np.arange(1, int(np.log2(maxfreqs))+1)

    # 1. Compute analytic and numerically integrated covariances.
    cov = dict(
        analytic=model.calculate_covariance(params, noise),
        integrated=np.array([
            model.integrate_covariance_from_analytic_spectrum(
                params, noise, nfreqs
            ) for nfreqs in nfreqs_trials
        ])
    )

    # 2. Plot numerically integrated covariances for different numbers of
    # sampled frequencies.
    pp.figure(figsize=(15, 15))

    nvars = len(model.vars)
    for i, j in itertools.combinations_with_replacement(range(nvars), 2):
        pp.subplot(nvars, nvars, j * nvars + i + 1)
        pp.axhline(cov['analytic'][i, j], color='k', label='analytic')
        pp.plot(
            nfreqs_trials,
            cov['integrated'][:, i, j],
            color='g',
            ls='--',
            label='integrated'
        )
        pp.xlim(nfreqs_trials[0], nfreqs_trials[-1])
        pp.yscale('log')
        pp.xscale('log')

        pp.ylabel('$\\Sigma_{%s%s}$' % (model.vars[i], model.vars[j]))
        pp.xlabel('$F$')

    # 3. Legend
    lastax = pp.gca()
    pp.subplot(nvars, nvars, nvars)
    pp.legend(*lastax.get_legend_handles_labels())
    pp.axis('off')

    # 4. Adjust and save plot.
    pp.subplots_adjust(wspace=0.5, hspace=0.5)

    pp.savefig('plots/covariance-test-integration.pdf')


def pool_cov(data):
    model, params, noise, ts = data
    return np.cov(
        model.simulate_linear(np.zeros(len(model.vars)), params, noise, ts),
        rowvar=1
    )


def plot_simulated(model, noise, params, maxtimesteps=2**12, trials=64):
    """
    Plot convergence test for covariance estimate from simulated data. Should
    line up with both the analytic covariance and the numerically integrated
    covariance. Saves a plot called plots/covariance-test-simulation.pdf.

    :param model (models.StochasticModel instance): model
    :param noise (np.ndarray): noise covariance
    :param params (dict): parameters for the model, keyed by SymPy symbols.
    :param maxtimesteps (int): maximum number of timesteps in simulation.
    """

    timesteps = 2**np.arange(10, int(np.log2(maxtimesteps)))
    evensteps = [t for t in np.log2(timesteps) if t % 2 == 0]

    nvars = len(model.vars)
    analytic_cov = model.calculate_covariance(params, noise)

    # 1. Calculate distribution of covariances for different series lengths.
    covariances = np.zeros((len(timesteps), trials, nvars, nvars))

    pool = multiprocessing.Pool()

    for i, ts in enumerate(timesteps):
        print 'Simulating %d timesteps.' % ts
        covariances[i] = pool.map(
            pool_cov,
            [(model, params, noise, ts)] * trials
        )

    # 2. Plot scatter of distribution of covariances for different lengths.
    pp.figure(figsize=(15, 15))

    for i, j in itertools.combinations_with_replacement(range(nvars), 2):
        pp.subplot(nvars, nvars, j * nvars + i + 1)

        y1 = np.percentile(covariances[:, :, i, j], 1, axis=1)
        y99 = np.percentile(covariances[:, :, i, j], 99, axis=1)
        ymean = np.mean(covariances[:, :, i, j], axis=1)
        y25 = np.percentile(covariances[:, :, i, j], 25, axis=1)
        y75 = np.percentile(covariances[:, :, i, j], 75, axis=1)

        pp.fill_between(timesteps, y1, y99, color='r', alpha=0.25)
        pp.fill_between(timesteps, y25, y75, color='r', alpha=0.25)
        pp.plot(timesteps, ymean, color='r', label='$\\text{estimate}$')
        pp.axhline(analytic_cov[i, j], color='k', label='$\\text{analytic}$')

        pp.xlim(min(timesteps), max(timesteps))
        pp.xscale('log')
        #pp.yscale('log')
        pp.ylabel('$\\hat{\\Sigma}_{%s%s}$' % (model.vars[i], model.vars[j]))
        pp.xlabel('$\\text{timesteps }(T)$')

        pp.xticks(evensteps, ['$2^{%d}$' % t for t in evensteps])

    # 3. Legend.
    lastax = pp.gca()
    pp.subplot(nvars, nvars, nvars)
    pp.legend(*lastax.get_legend_handles_labels())
    pp.axis('off')

    # 4. Adjust and save plot.
    pp.subplots_adjust(wspace=0.5, hspace=0.5)

    pp.savefig('plots/covariance-test-simulation.pdf')


def main():
    config = json.load(open(sys.argv[1]))

    # 1. Create the model from the loaded configuration.
    model = models.Parasitism.get_model(config['simulation']['model'])
    noise = np.array(config['simulation']['covariance'])
    params = {
        models.Parasitism.params[name]: value for name, value in
        config['simulation']['params'].iteritems()
    }

    # 2. Test convergence of numerically integrated cospectrum.
    plot_numerically_integrated(model, noise, params)

    # 3. Test convergence of covariance of simulated data.
    plot_simulated(model, noise, params, 2**20, 2**8)

if __name__ == '__main__':
    main()