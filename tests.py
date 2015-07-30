import unittest
import sympy
import numpy as np
import models.stochastic


def monotonic_convergence(l):
    """
    See if a list of values increases or decreases monotonically and if its
    derivative also decreases monotonically.

    :param l: list of values.
    :return: (bool) whether or not there is monotonic convergence.
    """

    diff = np.abs(np.diff(l))
    return np.all(np.less_equal(diff[:-1], diff[1:]))


class TestStochasticMethods(unittest.TestCase):
    x = sympy.symbols('x x_1 x_2')
    eps = sympy.symbols('eps eps_1 eps_2')

    u_var = 1.0
    model = models.stochastic.StochasticModel(
        [x[0]], [eps[0]], [0.5 * x[0] + 0.5 * eps[0]]
    )

    u_var2 = np.array([[1.0, 0.5], [0.5, 1.0]])
    model2 = models.stochastic.StochasticModel(
        [x[1], x[2]],
        [eps[1], eps[2]],
        [(x[1] + x[2] + eps[1]) / 3, (x[1] + x[2] + eps[2]) / 3]
    )

    def test_analytic_variance(self):
        # X_t = .5X_{t-1} + .5u, u ~ N(0, sigma_u)
        # sigma^2_x = (1/2)^2 sigma^2_x + (1/2)^2 sigma^2_u
        #   sigma^2_x = (1/4)sigma^2_x + (1/4)sigma^2_u
        #   (3/4)sigma^2_x = (1/4)sigma^2_u
        #   sigma^2_x = (1/3)sigma^2_u
        analytic_var_1 = self.model.calculate_covariance({}, [self.u_var])
        self.assertEqual(analytic_var_1, self.u_var / 3)

        # This one's a little more complicated.
        analytic_var_2 = self.model2.calculate_covariance({}, [self.u_var2])
        self.assertTrue(np.allclose(
            analytic_var_2, np.array([[8./45, 11./90], [11./90, 8./45]])
        ))

    def test_integrated_variance(self):
        # TODO: I need to test more complicated models to make sure these equal.
        # TODO:     Currently, I haven't really validated that the eigen-
        # TODO:     decomposition code actually works for all models . . .

        integrated_vars_1 = [
            self.model.integrate_covariance_from_analytic_spectrum(
                {}, [self.u_var], n**2
            ) for n in range(1, 16)
        ]

        # First, make sure this is monotonically converging.
        self.assertTrue(monotonic_convergence(integrated_vars_1))

        # Second, make sure it's reasonably close to the analytic result.
        # TODO: This "tolerance" value is totally arbitrary.
        analytic_var_1 = self.model.calculate_covariance({}, [self.u_var])
        diff_1 = np.abs(integrated_vars_1[-1] - analytic_var_1)
        tolerance = 2 ** 43
        self.assertTrue(diff_1 <= np.finfo(diff_1.dtype).eps * tolerance)

    def test_simulated_variance(self):
        # TODO: Stub. Need to find a suitable way to test accuracy /
        # TODO:     convergence of covariance of random simulations.
        pass

    def test_correlation(self):
        self.assertEqual(
            models.utilities.correlation(
                self.model.calculate_covariance({}, [self.u_var])
            ), 1.0
        )

        self.assertTrue(
            np.array_equal(
                models.utilities.correlation(
                    self.model2.calculate_covariance({}, [self.u_var2])
                ),
                [[1, 11./16], [11./16, 1]]
            )
        )

    def test_covariance_multiples(self):
        cov_base_1 = self.model.calculate_covariance({}, self.u_var)
        cov_base_2 = self.model2.calculate_covariance({}, self.u_var2)

        for mult in np.linspace(-2, 2, 9):
            cov_1 = self.model.calculate_covariance({}, self.u_var * mult)
            cov_2 = self.model2.calculate_covariance({}, self.u_var2 * mult)

            self.assertEqual(cov_1, cov_base_1 * mult)
            self.assertTrue(np.allclose(cov_2, cov_base_2 * mult))

if __name__ == '__main__':
    unittest.main()
