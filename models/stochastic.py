import string

import numpy as np
import sympy

import warnings
import scipy.signal

import utilities


class StochasticModel:
    """
    Describes a system of stochastic difference equations with a set of
    state variables, noise variables sampled from a multinormal distribution,
    and a matrix expression as a function of both. Variables are SymPy symbols,
    and the dynamics are expressed as a SymPy Matrix object.
    """

    def __init__(self, symvars, noises, equation, allvars=[], cachesize=10000):
        """
        Initialize a model object.
        :param symvars: list of SymPy symbols representing state variables.
        :param noises: list of SymPy symbols representing noise variables.
        :param equation: SymPy Matrix describing dynamics of the state
            variables. X_{t+1} = F(X_t)
        :param allvars:
        :param cachesize:
        """

        # TODO: Describe allvars, cachesize.

        self.allvars = allvars
        self.vars = sympy.Matrix(symvars)
        self.noises = sympy.Matrix(noises)
        self.stochastic = sympy.Matrix(equation)
        self.deterministic = self.stochastic.subs(
            dict(zip(noises, [0] * len(noises)))
        )

        self.equilibrium = None
        self.A = None
        self.B = None

        self.simulated = None
        self.simulated_linear = None

        # If lambdify_ss is set to True, get_cached_matrices will first convert
        # (and cache) the state space matrices as lambda functions, rather than
        # using sympy's matrix evaluation function.
        self.lambdify_ss = False
        self.lambdas = dict()

        # Cache for matrices evaluated for certain parameters. The cache is
        # indexed by a hash of the dictionary of parameters.
        self.cache = utilities.CacheDict(size_limit=cachesize)

    def solve_equilibrium(self):
        """Find the [first] non-trivial equilibrium point."""

        # TODO: This currently assumes there's one non-trivial equilibrium
        # point and that it's the second returned by SymPy's solve().
        # That's a dangerous assumption.

        self.equilibrium = sympy.solve(
            sympy.Eq(self.vars, self.deterministic),
            self.vars,
            dict=True
        )[-1]

        return self.equilibrium

    def linearize(self):
        """
        Create deterministic / stochastic transform matrices for linearization
        about the equilibrium point. Note that the equilibrium must be set
        before this is called.  This creates two matrices, m1 and q0. m1 is the
        deterministic transition matrix for the linear model and q0 is the
        noise-dependence matrix.
        """

        if self.equilibrium is None:
            self.solve_equilibrium()

        # Use equilibrium with 0 noise terms for substitutions.
        subs = dict(self.equilibrium.items())
        subs.update({noise: 0 for noise in self.noises})

        self.A = self.deterministic.jacobian(self.vars).subs(subs)
        self.B = self.stochastic.jacobian(self.noises).subs(subs)

    def lambdify(self, var, key, params):
        # Make sure the keys in params are ordered as they are in allvars, so
        # that consecutive calls to cached lambdified matrices will have the
        # same argument ordering.
        """

        :param var:
        :param key:
        :param params:
        :return:
        """

        param_keys = [k for k in self.allvars if k in params]
        param_vals = [params[k] for k in param_keys]

        if key not in self.lambdas:
            # Use sympy.utilities.lambdify to set up a lambda function to
            # quickly evaluate the model at each timestep. Lamdify just
            # evaluates a string of Python code, so first replace any special
            # LaTeX characters with _.
            trans = string.maketrans(r'{}\()^*[]', r'_________')
            fixed_keys = [str(k).translate(trans) for k in param_keys]
            subs = dict(zip(param_keys, fixed_keys))
            self.lambdas[key] = sympy.utilities.lambdify(
                fixed_keys, var.subs(subs)
            )

        return self.lambdas[key](*param_vals)

    def get_cached_matrices(self, params):
        """
        Cache m1/q0 matrices for the given parameter values to avoid
        recomputing each time we want to compute spectral properties for the
        same parameter values (e.g. computing the full spectral matrix at
        multiple frequencies).

        :param params: (dict) dictionary of parameter values with SymPy symbol
            keys.
        :return: (dict) dictionary with two keys, 'm1' and 'q0', each
            containing a numpy array for the matrix evaluated at the parameter
            values.
        """

        if self.A is None or self.B is None:
            self.linearize()

        phash = hash(frozenset(params.items()))
        self.cache.setdefault(phash, {})
        cached = self.cache[phash]

        if 'A' not in cached:
            if self.lambdify_ss:
                cached['A'] = self.lambdify(self.A, 'A', params)
            else:
                cached['A'] = utilities.eval_matrix(self.A.subs(params))

        if 'B' not in cached:
            if self.lambdify_ss:
                cached['B'] = self.lambdify(self.B, 'B', params)
            else:
                cached['B'] = utilities.eval_matrix(self.B.subs(params))

        return cached

    def integrate_covariance_from_analytic_spectrum(
        self,
        params,
        inputcov,
        nfreqs=1024
    ):
        """
        Compute numerically integrated covariance for the system state over
        a specified number of frequencies.

        :param params: (dict) model parameter values keyed by sympy symbols.
        :param inputcov: (np.narray) noise covariance.
        :param nfreqs: (int) number of frequencies over which to sum (default:
            1024).
        :return: (np.narray) covariance matrix.
        """

        return np.sum([
            np.abs(self.calculate_spectrum(params, inputcov, f))
            for f in np.linspace(-0.5, 0.5, nfreqs)
        ], axis=0) / nfreqs

    def state_space(self, params):
        """
        Return the (A, B, C, D) state space representation for the model for
        use with scipy.signal methods.

        :param params: (dict) model parameters keyed by sympy symbols.
        :return: (np.array, np.array, np.array, np.array) state space
            representation matrices.
        """

        cached = self.get_cached_matrices(params)
        a, b = [np.matrix(cached[k]) for k in 'A', 'B']

        return a, b, np.eye(len(a)), np.zeros(a.shape)

    def zeros_and_poles(self, params):
        """
        Return the zeros and poles for the model.

        :param params: (dict) model parameters keyed by sympy symbols.
        :return: (np.array, np.array) lists of zeros and poles.
        """
        num, den = scipy.signal.ss2tf(*self.state_space(params))
        zpks = [scipy.signal.tf2zpk(n, den) for n in num]

        return zpks

    def transfer_function(self, params):
        cached = self.get_cached_matrices(params)
        a, b = cached['A'], cached['B']

        return lambda z: np.dot(
            np.linalg.inv(z * np.eye(len(self.vars)) - a),
            b
        )

    def calculate_covariance(self, params, inputcov):
        """
        Calculate the covariance (autocovariance with lag zero) for the model.
        :param params: (dict) parameter values with SymPy symbol keys.
        :param inputcov: (np.array) covariance of the noise.
        :return: (np.array) covariance matrix.
        """

        cached = self.get_cached_matrices(params)
        a, b = cached['A'], cached['B']

        # R(0) = M1 R(0) M1' + Q0 Sigma Q0'
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            result = np.real(
                utilities.solve_axatc(a, np.dot(np.dot(b, inputcov),  b.T))
            )
        else:
            warnings.warn(' '.join([
                'A and/or B matrix contained inf or NaN values. Returning NaN',
                'covariance matrix. Input parameters: %s.' % str(params)
            ]))
            result = np.full_like(a, np.nan)

        return result

    def calculate_eigenvalues(self, params):
        """
        Calculate the dominant frequency of oscillation for the linearized
        model. This will be the same for all cross-spectra.

        :param params: (dict) dictionary of parameter values with SymPy symbol
            keys.
        :return: (float) frequency of the system's oscillation.
        """

        cached = self.get_cached_matrices(params)
        return np.linalg.eig(cached['A'])

    def calculate_spectrum(self, params, inputcov, v=0):
        """
        Calculate the spectral matrix according to the model linearized around
        its equilibrium. Note that linearize() must be called before this.

        :param params: (dict) free parameters to the model (excluding
            state/noise). Keys are SymPy symbols.
        :param inputcov: (np.array) covariance of the noise, assuming each
            noise parameter is a dimension of a multivariate normal, with
            dimensions ordered according to self.noises.
        :param v: (float) frequency at which to evaluate the spectral matrix
            (0 to 0.5).
        :return: NxN matrix of co-spectra, where N is the number of state
            variables.
        """

        r = self.transfer_function(params)(np.exp(-2j * np.pi * v))
        return np.dot(r, np.dot(inputcov, np.conj(r.T)))

    def simulate(self, initial, params, noise, timesteps=2**10):
        """
        Start from an initial point and simulate the model with sampled noise.
        :param initial: (list) Initial values for the state variables, ordered
            according to self.vars.
        :param params: (dict) free parameters to the model (excluding state/
            noise). Keys are SymPy symbols.
        :param noise: (np.array) covariance of the noise, assuming each
            noise parameter is a dimension of a multivariate normal, with
            dimensions ordered according to self.noises.
        :param timesteps: (int) number of timesteps to simulate.
        :return: NxT matrix of state vectors, where N is the number of
            variables and T is the number of timesteps.
        """

        nstates = len(initial)

        # Begin the model at the equilibrium.
        params = dict(params.items())
        params.update({noise_var: 0 for noise_var in self.noises})
        params.update({k: self.equilibrium[k] for k in self.vars})
        param_keys = params.keys()

        # Use sympy.utilities.lambdify to set up a lambda function to quickly
        # evaluate the model at each timestep. Lamdify just evaluates a string
        # of Python code, so first replace any special LaTeX characters with _.
        trans = string.maketrans(r'{}\()^*[]', r'_________')
        fixed_keys = [str(k).translate(trans) for k in param_keys]
        subs = dict(zip(param_keys, fixed_keys))

        param_inds = {param_keys[i]: i for i in range(len(param_keys))}
        param_vals = [params[k] for k in param_keys]

        f = sympy.utilities.lambdify(fixed_keys, self.stochastic.subs(subs))

        # Simulate the model.
        rnoise_mean = np.zeros((len(noise),))
        self.simulated = np.zeros((timesteps, len(initial)))
        self.simulated[0] = initial

        for step in range(1, timesteps):
            # Generate random noise with the given covariance.
            rnoise = np.random.multivariate_normal(rnoise_mean, noise)

            # Fill the model parameters with the output of the last iteration.
            for i, noise_var in enumerate(self.noises):
                param_vals[param_inds[noise_var]] = rnoise[i]

            for i, state_var in enumerate(self.vars):
                param_vals[param_inds[state_var]] = self.simulated[step - 1, i]

            # Evaluate the model at this timestep.
            sim = f(*param_vals)

            self.simulated[step] = np.reshape(sim, (nstates,))

        return self.simulated.T

    def simulate_linear(self, initial, params, noise, timesteps=1000):
        """
        Start from an initial point and simulate the model linearized about
        the equilibrium with sampled noise.

        :param initial: (list) Initial values for the state variablees, ordered
            according to self.vars.
        :param params: (dict) free parameters to the model (excluding state/
            noise). Keys are SymPy symbols.
        :param noise: (np.array) Covariance of the noise, assuming each
            noise parameter is a dimension of a multivariate normal, with
            dimensions ordered according to self.noises.
        :param timesteps: (int) number of timesteps to simulate.
        :return: NxT matrix of state vectors, where N is the number of
            variables and T is the number of timesteps.
        """

        cached = self.get_cached_matrices(params)
        a, b = cached['A'], cached['B']

        # System tuple: A, B, C, D, dt
        if len(a) > 1:
            u = np.random.multivariate_normal(
                np.zeros_like(noise), noise, timesteps
            )
        else:
            u = np.random.normal(0, np.sqrt(float(noise[0, 0])), (timesteps, 1))

        _, yout, xout = scipy.signal.dlsim(
            (a, b, np.eye(len(a)), np.zeros(a.shape), 1), u, x0=initial
        )

        self.simulated_linear = yout.T

        return self.simulated_linear


