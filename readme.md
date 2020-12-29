# parasynchrony

Some tools for modeling population synchrony in host-parasitoid models. 

- models.py: This module defines a StochasticModel, which uses
    SymPy to linearize a model expressed as a nonlinear difference equation
    around a stationary equilibrium point and compute its spectrum. The class
    Parasitism defines a few common host-parasitoid models as StochasticModels. 
- test-cospectrum.py: some over-engineered code to verify the analytic results 
    for a linearized model's spectrum by simulating the linear model for
    increasing durations seeing that spectral estimates converge.
- test-covariance.py: same thing, but for covariance. Convergence of
    a) numerically integrated cospectrum for increasing number of sampled
        frequencies, and
    b) covariance estimate of simulated data for increasing number of timesteps.
- fraction.py: decompose each model into the fraction of synchrony for which
    each exogenous source of synchrony is responsible.
- variants.py: plot analytic spectra for a variety of model variants.
