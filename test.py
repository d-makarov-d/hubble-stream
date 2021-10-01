import unittest
import numpy as np
import matplotlib.pyplot as plt

from eq_10 import SolverOptions, Solver


class Test(unittest.TestCase):
    def test_exact_solution(self):
        omega_m = (1e-10, 0.6)
        n_points = 500
        H0 = 66.9

        om = np.linspace(omega_m[0], omega_m[1], n_points)
        ol = 1 - om

        a = np.zeros_like(om)

        for i, o in enumerate(om):
            a[i] = Solver(SolverOptions(o)).a(0)

        plt.plot(om, a)
        plt.show()
