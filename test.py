import unittest

import numpy as np
import matplotlib.pyplot as plt

from eq_10 import Solver
from physics import PhysicOptions, Point, Halo, Empty


class Test(unittest.TestCase):
    def test_exact_solution(self):
        omega_m = (1e-10, 0.6)
        n_points = 100

        om = np.linspace(omega_m[0], omega_m[1], n_points)

        a = np.zeros_like(om)

        for i, o in enumerate(om):
            a[i] = Solver(PhysicOptions(o)).a(0)

        plt.figure()
        plt.plot(om, a)
        plt.show()

    def test_profiles(self):
        omega_m = 0.3
        omega_l = 1 - omega_m
        r0 = np.linspace(0.5, 3, 1000)
        L0 = 1

        options = PhysicOptions(omega_m, H0=73)
        solver = Solver(options)
        profiles = [
            (Point(options).model(L0), 'point',),
            (Halo(options).model(L0), 'halo',),
            (Empty(options).model(L0), 'empty',),
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        for profile, name in profiles:
            sigma = profile(r0)
            alpha = sigma / omega_l
            h = solver.h(alpha)
            v = h * r0

            ax1.plot(r0, v, label=name)
            ax1.set_xlabel('r0')
            ax1.set_ylabel('v')
            ax1.legend()
            ax2.plot(r0, h, label=name)
            ax2.set_xlabel('r0')
            ax2.set_ylabel('h')
            ax2.legend()
        plt.show()

    def test_opposite_solution(self):
        omega_m = 0.3
        r0 = np.linspace(0.5, 3, 100)
        L0 = 1

        options = PhysicOptions(omega_m, H0=73)
        solver = Solver(options)

        profile = Point(options).model(L0)
        sigma = profile(r0)
        alpha = sigma / options.omega_l_0
        h = solver.h(alpha)
        a = solver.a(h)

        self.assertTrue(max(alpha - a) < np.finfo(float).eps ** 0.5)

