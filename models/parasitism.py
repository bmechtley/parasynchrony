"""
models/parasitism.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Host-parasitoid StochasticModels. Includes:
    ar1: first-order autoregressive process
    nb: Nicholson-Bailey (unstable)
    nbd: Negative-Binomial Nicholson-Bailey (stable)
    get_model: Any of the above with even dispersal between multiple patches.
"""

import sys
import sympy
import stochastic

symbols = dict(
    a=sympy.Symbol('a', positive=True),
    r=sympy.Symbol('\lambda', positive=True),
    c=sympy.Symbol('c', positive=True),
    k=sympy.Symbol('k', positive=True),
    mh=sympy.Symbol('\mu_H', positive=True),
    mp=sympy.Symbol('\mu_P', positive=True),
    h=[sympy.Symbol('H')] + list(sympy.symbols('H^{((1:3))}')),
    p=[sympy.Symbol('P')] + list(sympy.symbols('P^{((1:3))}')),
    eh=[sympy.Symbol('\epsilon_h')] + list(
        sympy.symbols('\epsilon^{((1:3))}_h')
    ),
    ep=[sympy.Symbol('\epsilon_p')] + list(
        sympy.symbols('\epsilon^{((1:3))}_p')
    ),
    alpha=sympy.Symbol(r'\alpha'),
    x=sympy.Symbol('x'),
    e=sympy.Symbol(r'\epsilon')
)


def sym_params(params):
    return {symbols[k]: v for k, v in params.iteritems()}


def ar1():
    """
    AR-1 autoregressive model for one state variable,
        x_t = \alpha x_{t-1} + e
    :return: StochasticModel for AR1 process.
    """

    a, x, e = [symbols[v] for v in ['alpha', 'x', 'e']]
    return stochastic.StochasticModel([x], [e], [a * x + e])


def nb():
    """
    Single-patch Nicholson-Bailey model with no regularization. Note: this
    has an unstable equilibrium.
        H_t = \lambda H_{t-1} exp(-a P_{t-1}) exp(e_H)
        P_t = c H_{t-1} (1 - exp(-a P_{t-1})) exp(e_P)
    :return: StochasticModel for Nicholson-Bailey process.
    """
    h, p, eh, ep, r, a, c = [symbols[v] for v in [
        'h', 'p', 'eh', 'ep', 'r', 'a', 'c'
    ]]

    return stochastic.StochasticModel(
        [h[0], p[0]],
        [eh[0], ep[0]],
        [
            r * h[0] * sympy.exp(-a * p[0]) * sympy.exp(eh[0]),
            c * h[0] * (1 - sympy.exp(-a * p[0])) * sympy.exp(ep[0])
        ]
    )


def nbd():
    """
    Single-patch Nicholson-Bailey model with negative binomial functional
    response.
        H_t = \lambda H_{t-1} (1 + a \frac{P_{t-1}}{K})^{-k} exp(e_H)
        P_t = c H_{t-1} (1 - (1 + a \frac{P_{t-1}}{K})^{-k}) exp(e_P)
    :return: StochasticModel for Nicholson-Bailey process with negative
        binomial functional response.
    """
    h, p, eh, ep, r, a, c, k = [symbols[v] for v in [
        'h', 'p', 'eh', 'ep', 'r', 'a', 'c', 'k'
    ]]

    return stochastic.StochasticModel(
        [h[0], p[0]],
        [eh[0], ep[0]],
        [
            r * h[0] * (1 + a * p[0] / k) ** -k * sympy.exp(eh[0]),
            c * h[0] * (1 - (1 + a * p[0] / k) ** -k) * sympy.exp(ep[0])
        ]
    )


def get_model(modelstr):
    """
    Get a model by name. For each constructor in Parasitism (e.g. ar1, nb,
    nbd), the name can be taken along with an integer parameter indicating
    the number of patches to use for a global dispersal matrix parameterized
    by m_H and m_P, where each is the probability that a member of the
    population will move to another patch. Dispersal is global and
    symmetric, so each patch obtains equal migration from the others.

    :param modelstr (str): String indicating the model and number of
        patches as "model(N)", e.g. "nb(1)" or "nbd(2)". If no parameter
        is specified (i.e. just "model"), a single-patch model will be
        returned.
    :return: StochasticModel for the specified multipatch model.
    """

    h, p, eh, ep, r, a, c, k, mp, mh = [symbols[v] for v in [
        'h', 'p', 'eh', 'ep', 'r', 'a', 'c', 'k', 'mp', 'mh'
    ]]

    # Tokenize model string to obtain the specified model.
    tokens = modelstr.split('(')
    refmodel = getattr(sys.modules[__name__], tokens[0])()
    refmodel.solve_equilibrium()
    refmodel.linearize()

    if len(tokens) < 2:
        # If number of patches is not specified, return the one-patch model.
        return refmodel
    else:
        # Make a multipatch model from the reference model with global
        # dispersal.

        params = tokens[1].rstrip(')').split(',')
        npatches = int(params[0])
        patches = range(npatches)

        migration = params[1] if len(params) > 1 else 'global'

        # migrationmat will be the transition matrix applied to the
        # difference equations for population dynamics in each patch.
        # For global dispersal, it will have 1-m in the diagonal and m / N
        # in the off diagonals such that hosts only migrate to other hosts
        # and parasitoids only migrate to other parasitoids.
        #
        # TODO: This will NOT work, e.g. with AR1 or any non host-
        # TODO:     parasitoid model or multispecies model. Only for models
        # TODO:     where there are one host and one parasitoid per patch,
        # TODO:     and hosts precede parasitoids in the list of state
        # TODO:     variables.
        migrationmat = sympy.zeros(npatches * 2)

        if migration == 'global':
            for row in range(npatches * 2):
                for col in range(npatches * 2):
                    m = mh if row < npatches else mp

                    if row == col:
                        migrationmat[row, col] = 1-m
                    elif row < npatches and col < npatches:
                        # Host migration.
                        migrationmat[row, col] = m / npatches
                    elif row >= npatches and col >= npatches:
                        # Parasitoid migration.
                        migrationmat[row, col] = m / npatches

        # Ordinary difference equations to which migration matrix will be
        # applied.
        equations = sympy.Matrix([
            item for sublist in [
                [
                    refmodel.stochastic[i].subs({
                        h[0]: h[j+1],
                        p[0]: p[j+1],
                        eh[0]: eh[j+1],
                        ep[0]: ep[j+1]
                    }) for j in patches
                ] for i in range(2)
            ] for item in sublist
        ])

        # Create the model.
        multipatch = stochastic.StochasticModel(
            [h[i+1] for i in patches] + [p[i+1] for i in patches],
            [eh[i+1] for i in patches] + [ep[i+1] for i in patches],
            migrationmat * equations
        )

        # Solve for the equilibrium. Assuming global dispersal, the
        # equilibrium should be uniform across all patches and not depend
        # on the migration parameters.

        if refmodel.equilibrium is not None:
            multipatch.equilibrium = {
                h[i+1]: refmodel.equilibrium[h[0]] for i in patches
            }

            multipatch.equilibrium.update({
                p[i+1]: refmodel.equilibrium[p[0]] for i in patches
            })

        # Linearize the model.
        multipatch.linearize()

        return multipatch
