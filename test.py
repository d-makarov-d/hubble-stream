import unittest

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import optimize
from astropy import units as u

from eq_10 import Solver
from observation_model import VelocityField, Galaxy
from physics import PhysicOptions, Point, Halo, Empty
from util import Vector, _cartesian_to_spherical, draw_vectors, least_square_xy, J
from files import load_vizier


def rand_sph(r: float) -> Vector:
    _r = random.uniform(0, r) ** 0.5 * r ** 0.5
    _lat = random.uniform(-np.pi / 2, np.pi / 2)
    _lon = random.uniform(0, 2 * np.pi)
    return Vector.get_sph([_r, _lat, _lon])


def _rand_norm(d: float):
    return Vector.get_cart([
        random.normalvariate(0, d),
        random.normalvariate(0, d),
        random.normalvariate(0, d),
    ])


class ZeroField(VelocityField):
    def field(self, coord: Vector, *params) -> Vector:
        return Vector.get_cart([0, 0, 0])


class LinearField(VelocityField):
    def field(self, coord: Vector, h0) -> Vector:
        return Vector.get_sph([h0 * (coord.r - 1), coord.lat, coord.lon])


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


class TestFiles(unittest.TestCase):
    def test_load_vizier(self):
        gals = load_vizier(['data/table2.dat'], ['data/vel1.dat'])
        coords = [gal.coordinates for gal in gals.values()]
        vels = [gal.velocity for gal in gals.values()]
        vels = [Vector.get_sph([v, c.lat, c.lon]) for v, c in zip(vels, coords)]
        draw_vectors(coords, vels, norm_len=30)

    def _test_save_matlab(self):
        gals = load_vizier(['data/table2.dat'], ['data/vel1.dat'])
        from scipy.io import savemat
        names = [n for n in gals.keys()]
        vels = [gal.velocity for gal in gals.values()]
        dists = [gal.coordinates.r for gal in gals.values()]
        de = [gal.coordinates.lat for gal in gals.values()]
        ra = [gal.coordinates.lon for gal in gals.values()]
        data = {
            'names': names,
            'velocities': vels,
            'distance': dists,
            'DERad': de,
            'RARad': ra
        }
        savemat('data/galaxies.mat', data)


class VectorTest(unittest.TestCase):
    def test_property_assignment(self):
        v = Vector.get_cart([1, 2, 3])
        v.cart = [3, 3, 3]
        sph_new = _cartesian_to_spherical([3, 3, 3])
        self.assertTrue(max(abs(sph_new - v.sph)) < np.finfo(float).eps)

    def test_property_assignment_1(self):
        v = Vector.get_cart([1, 2, 3])
        v.x = 2
        sph_new = _cartesian_to_spherical([2, 2, 3])
        self.assertTrue(max(abs(sph_new - v.sph)) < np.finfo(float).eps)
        self.assertTrue(max(abs(np.array([2, 2, 3]) - v.cart)) < np.finfo(float).eps)

    def test_property_assignment_2(self):
        v = Vector.get_cart([1, 2, 3])
        v.x += 1
        sph_new = _cartesian_to_spherical([2, 2, 3])
        self.assertTrue(max(abs(sph_new - v.sph)) < np.finfo(float).eps)
        self.assertTrue(max(abs(np.array([2, 2, 3]) - v.cart)) < np.finfo(float).eps)


class ModelTest(unittest.TestCase):
    def test_dot_field(self):
        n_points = 100
        err = 0.1
        coords = list(_rand_norm(10) for i in range(n_points))
        vels = list(_rand_norm(1) for i in range(n_points))
        mw_vel = vels.pop() * 100
        mw_coord = coords.pop()
        observations = [
            Vector.unit(mw_coord - coord).dot(vel - mw_vel) * np.random.uniform(1-err, 1+err)
            for coord, vel in zip(coords, vels)
        ]
        model = ZeroField()

        def f(v: np.ndarray) -> float:
            velocity = Vector.get_cart(v)
            acc = 0
            for coord, vel, obs in zip(coords, vels, observations):
                acc += (obs - model.velocity(velocity, mw_coord - coord)) ** 2
            return acc

        vel = optimize.minimize(f, [0, 0, 0])
        print(vel.x)
        print(mw_vel)
        # self.assertTrue((abs(Vector.get_cart(vel.x) - mw_vel) / abs(mw_vel)) < (err / 3))
        self.assertTrue((mw_vel - Vector.get_cart(vel.x)).r < 1)

    def test_self_velocity(self):
        n_points = 100
        err = 0.
        coords = list(_rand_norm(10) for i in range(n_points))
        vels = list(_rand_norm(1) for i in range(n_points))
        mw_vel = vels.pop() * 100
        mw_coord = coords.pop()
        obs_vels = [
            Vector.unit(mw_coord - coord).dot(vel - mw_vel) #  * np.random.uniform(1 - err, 1 + err)
            for coord, vel in zip(coords, vels)
        ]
        observations = [Galaxy(mw_coord - coord, vel) for coord, vel in zip(coords, obs_vels)]
        res, _ = VelocityField.observer_velocity(observations)

        print(res)
        print(mw_vel)

        self.assertTrue((res - mw_vel).r < 0.5)

    def test_simple_model(self):
        r = 10.
        r_min = 0.3
        n0 = 500
        h0 = 1
        model = LinearField()

        coord_arr = np.random.uniform(3 * [-r], 3 * [r], (n0, 3))
        coords = [Vector(row) for row in coord_arr]
        coords = tuple(filter(lambda v: r_min < v.r < r, coords))
        vels = tuple(model.field(c, h0=h0) for c in coords)

        sun_coords = Vector(np.random.uniform(2, 3, 3))
        sun_vel = model.field(sun_coords, h0=h0) * Vector.unit(Vector.get_cart(np.random.uniform(-1, 1, 3))) * 10

        gals = tuple(
            Galaxy(coord - sun_coords, Vector.unit(coord - sun_coords).dot(vel - sun_vel))
            for coord, vel in zip(coords, vels)
        )

        res_v, res_r, p = model.fit_model(gals)
        print(f"R0:\n\texpected: {sun_coords}\n\tfound:    {res_r}")
        print(f"V0:\n\texpected: {sun_vel}\n\tfound:    {res_v}")
        print(f"H0:\n\texpected: {h0}\n\tfound:    {p[0]}")

        self.assertTrue((res_v - sun_vel).r < np.finfo(float).eps ** 0.5)
        self.assertTrue((res_r - sun_coords).r < np.finfo(float).eps ** 0.5)

    def test_two_groups(self):
        gals = load_vizier(['data/table2.dat'], ['data/vel1.dat'])
        # divide in two equal groups by distance
        gals = sorted(gals.values(), key=lambda g: g.coordinates.r)
        m = len(gals) // 2
        group1 = gals[:m]
        group2 = gals[m:]

        model = ZeroField()

        res_v, dv = model.observer_velocity(gals)
        res_v1, dv1 = model.observer_velocity(group1)
        res_v2, dv2 = model.observer_velocity(group2)

        print(f"V0:\n\t"
              f"group close: {res_v1} +- {dv1}\n\t"
              f"group far:   {res_v2} +- {dv2}\n\t"
              f"all:         {res_v} +- {dv}")

        print('In Galactic coordinates')
        print('Close group')
        print(res_v1.coords_icrs(u.km / u.s).galactic)
        print()
        print('Far group')
        print(res_v2.coords_icrs(u.km / u.s).galactic)
        print()
        print('Group all')
        print(res_v.coords_icrs(u.km / u.s).galactic)
        print()

        def cart_to_gal(cart):
            v = Vector.get_cart(cart)
            gal = v.coords_icrs(u.km / u.s).galactic
            return np.array([gal.l.radian, gal.b.radian, gal.distance.value])

        for r, d in zip((res_v1, res_v2, res_v), (dv1, dv2, dv)):
            jac = J(np.array(r.cart), cart_to_gal)
            err = np.matmul(np.matmul(jac, d.cart), jac.T)
            err[:2] = err[:2] / np.pi * 180
            print(err)
