"""
models/__init__.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Package for various stochastic difference equation models. In addition to a few
helper functions (mostly involved with plotting simulated results), this package
defines a StochasticModel class which is a generalized class for stochastic
difference equations, allowing the automatic analytic linearization of
sufficiently simple nonlinear models using SymPy. Additionally, the class
Parasitism contains several class methods for constructing prefab
StochasticModels, e.g. stochastic AR-1 processes (ar1), Nicholson-Bailey (nb),
and Nicholson-Bailey with negative binomial functional response (nbd) as well as
a generator that can use any of these single-patch models to create a
multipatch model with global dispersal.

stochastic: defines StochasticModel, the base stochastic difference eq model.
utilities: various helper utilities for the package
parasitism: defines several multipatch host-parasitoid StochasticModels.
plotting: some helper methods for plotting StochasticModel spectra etc.
"""


from stochastic import *
import utilities
import parasitism
import plotting