import unittest

import numpy as np
from scipy import optimize
from util import least_square_xy

from eq_10 import Solver
from physics import PhysicOptions, VelocityField, Point
from files import load_leda
from util import Vector

class ZeroField(VelocityField):
    def field(self, coord: Vector) -> Vector:
        return Vector.get_cart([0, 0, 0])

class ReferenceField(VelocityField):
    def field(self, coord: Vector, h) -> Vector:
        h *= int(coord.r > 1.5)
        return Vector.get_sph([h * coord.r, coord.lat, coord.lon])

    def fit_model(self, galaxies):
        obs_vels = np.array(tuple(gal.velocity for gal in galaxies))

        def f(v):
            vel = Vector.get_cart(v[:3])
            r = Vector.get_cart([0, 0, 0])
            p = v[3:]
            model_vels = np.zeros_like(obs_vels)
            for i, gal in enumerate(galaxies):
                model_vels[i] = self.velocity(r, vel, gal.coordinates, *p)
                #model_vels[i] = -np.sum(np.array(Vector.unit(gal.coordinates).cart) * np.array(vel.cart))

            return obs_vels - model_vels

        res = optimize.least_squares(f, [0, 0, 0, 1])
        res_v = Vector.get_cart(res.x[:3])

        return res_v, res.x[3:]

class Test(unittest.TestCase):
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

    def test_laq(self):
        fun = lambda x: 3 * x**2 + 12
        x = np.linspace(0, 3, 100)
        y = fun(x)
        y += np.random.normal(0, y * 0.1)

        f = lambda p: y - (p[0] * x**2 + p[1])

        res = optimize.least_squares(f, [1, 0])
        expected_res = res.x
        cov = np.linalg.inv(res.jac.T.dot(res.jac))
        var = np.sqrt(np.diagonal(cov))
        sigma = (sum(res.fun ** 2) / len(res.fun)) ** 0.5
        expected_err = var * sigma

        res, err = least_square_xy(f, [1, 0])

        print((expected_res, expected_err))
        print((res, err))

    def test_reference(self):
        """Compare new implementation to the reference implementation values"""
        gals = load_leda(['../data/reference.dat'], ra='l', dec='b', dist='Dist', vel='Vh').values()
        ref_verial = [92.7, -2.3, 333]  # l, b, v
        ref_unverial = [91.4, -0.3, 346, 66]  # l, b, v, h

        gals_verial = tuple(filter(lambda g: g.coordinates.r < 1.5, gals))

        z_field = ZeroField()
        v, dv = z_field.observer_velocity(gals_verial)
        l = v.lon / np.pi * 180
        b = v.lat / np.pi * 180
        self.assertEqual(ref_verial, [np.round(l, 1), np.round(b, 1), np.round(v.r)])

        r_field = ReferenceField()
        v, h = r_field.fit_model(gals)
        l = v.lon / np.pi * 180
        b = v.lat / np.pi * 180
        self.assertEqual(ref_unverial, [np.round(l, 1), np.round(b, 1), np.round(v.r), np.round(h)])
