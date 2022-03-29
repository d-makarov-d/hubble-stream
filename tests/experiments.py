import unittest

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import optimize
from astropy import units as u

from eq_10 import Solver
from util import Vector, J
from observation_model import VelocityField, Galaxy
from physics import PhysicOptions, DenseDistField, DenseDistMassMwM31, TwoMassDenseDistField
from files import load_vizier, load_leda


class ZeroField(VelocityField):
    def field(self, coord: Vector) -> Vector:
        return Vector.get_cart([0, 0, 0])


class LinearField(VelocityField):
    def field(self, coord: Vector, h0) -> Vector:
        return Vector.get_sph([h0 * (coord.r - 1), coord.lat, coord.lon])

class LinearFieldBiased(VelocityField):
    def field(self, coord: Vector, a, b) -> Vector:
        return Vector.get_sph([a * coord.r + b, coord.lat, coord.lon])


def _rand_norm(d: float):
    return Vector.get_cart([
        random.normalvariate(0, d),
        random.normalvariate(0, d),
        random.normalvariate(0, d),
    ])


class Test(unittest.TestCase):
    def test_vector_field(self):
        field = LinearField()
        h0 = 2
        M = Vector([1, -3, 3])  # Milky way coord
        T = Vector([-2, 6, 2])  # Observed galaxy
        v0 = Vector.get_sph([h0 * (M.r - 1), M.lat, M.lon])  # MW velocity
        vt = Vector.get_sph([h0 * (T.r - 1), T.lat, T.lon])  # Observed galaxy velocity

        self.assertEqual(field.field(M, h0), v0)
        self.assertEqual(field.field(T, h0), vt)

        r = T - M  # from MW to object
        v_obs_expected = Vector.unit(r).dot(vt - v0)

        v_obs = field.velocity(M, v0, r, h0)

        print((v_obs_expected, v_obs))
        self.assertAlmostEqual(v_obs_expected, v_obs)

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
                acc += (obs - model.velocity(mw_coord, velocity, mw_coord - coord)) ** 2
            return acc

        vel = optimize.minimize(f, [0, 0, 0])
        print(vel.x)
        print(mw_vel)
        print((mw_vel - Vector.get_cart(vel.x)).r)
        # self.assertTrue((abs(Vector.get_cart(vel.x) - mw_vel) / abs(mw_vel)) < (err / 3))
        self.assertTrue((mw_vel - Vector.get_cart(vel.x)).r < 4)

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
        sun_vel = model.field(sun_coords, h0=h0) * Vector.unit(Vector.get_cart(np.random.uniform(-1, 1, 3))) * 100

        gals = tuple(
            Galaxy(coord - sun_coords, Vector.unit(coord - sun_coords).dot(vel - sun_vel))
            for coord, vel in zip(coords, vels)
        )

        res_v, res_r, p, _ = model.fit_model(gals)
        print(f"R0:\n\texpected: {sun_coords}\n\tfound:    {res_r}")
        print(f"V0:\n\texpected: {sun_vel}\n\tfound:    {res_v}")
        print(f"H0:\n\texpected: {h0}\n\tfound:    {p[0]}")

        self.assertTrue((res_v - sun_vel).r < np.finfo(float).eps ** 0.5)
        self.assertTrue((res_r - sun_coords).r < np.finfo(float).eps ** 0.5)

    def test_two_groups(self):
        gals = load_vizier(['../data/table2.dat'], ['../data/vel1.dat'])
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

    def test_mass_mw(self):
        gals = load_leda(['../data/lv_all.dat'], vel='Vh')
        mw = gals['Milky Way']
        andromeda = gals['MESSIER031']
        gals.pop('Milky Way')
        gals.pop('MESSIER031')
        options = PhysicOptions(0.3, H0=73)
        v_apex = options.apex
        for gal in gals.values():
            gal.velocity = gal.velocity + v_apex.dot(Vector.unit(gal.coordinates))

        selected_gals = [gals['Eridanus 2'], gals['LeoT'], gals['Phoenix']]
        solver = Solver(options)
        _, ax = plt.subplots(1, 2)
        ax_u_r = ax[0]
        ax_m_r = ax[1]
        ax_u_r.set_title('U(R)')
        ax_m_r.set_title('Mm(R)')
        masses = []
        for gal in selected_gals:
            d = gal.coordinates
            r = d - mw.coordinates
            # u = (gal.velocity + v_apex.dot(d) / abs(d)) * (abs(r) * abs(d)) / (r.dot(d))
            v = abs(d - mw.coordinates) * abs(d) / (abs(d) ** 2 - mw.coordinates.dot(d)) * gal.velocity
            ax_u_r.scatter(r.r, v)

            r_mpc = r.r
            h = v / r_mpc
            a = solver.a(h)
            Mm = a * options.omega_l_0 * 4 * np.pi * r_mpc ** 3. * options.rho0 / 3.
            masses.append(Mm)
            ax_m_r.scatter(r.r, Mm)

        plt.show()
        print("Mass of MW: %s +- %s" % (
            np.format_float_scientific(np.mean(masses)),
            np.format_float_scientific(np.std(masses))
        ))

    def test_mass_m31(self):
        gals = load_leda(['../data/lv_all.dat'], vel='Vh')
        mw = gals['Milky Way']
        andromeda = gals['MESSIER031']
        gals.pop('Milky Way')
        gals.pop('MESSIER031')
        options = PhysicOptions(0.3, H0=73)
        v_apex = options.apex
        for gal in gals.values():
            gal.velocity = gal.velocity + v_apex.dot(Vector.unit(gal.coordinates))

        selected_gals = [gals['IC1613'], gals['And XVIII'], gals['Pegasus']]
        selected_gals = dict(
            (k, v) for k, v in gals.items()
            if (v.coordinates - andromeda.coordinates).r < 0.58 and (v.coordinates - mw.coordinates).r > (v.coordinates - andromeda.coordinates).r
        )
        solver = Solver(options)
        _, ax = plt.subplots(1, 2)
        ax_u_r = ax[0]
        ax_m_r = ax[1]
        ax_u_r.set_title('U(R)')
        ax_m_r.set_title('Mm(R)')
        masses = []
        for k, gal in selected_gals.items():
            d = gal.coordinates
            dm31 = andromeda.coordinates
            r = d - dm31
            D = dm31 - mw.coordinates
            v_gal = gal.velocity
            v_m31 = andromeda.velocity
            print([abs(r) * abs(d) / r.dot(d), D.dot(d) / D.dot(dm31) * abs(dm31) / abs(d)])
            v = abs(r) * abs(d) / r.dot(d) * (v_gal - v_m31 * D.dot(d) / D.dot(dm31) * abs(dm31) / abs(d))

            try:
                r_mpc = r.r
                h = v / r_mpc
                a = solver.a(h)
                Mm = a * options.omega_l_0 * 4 * np.pi * r_mpc ** 3. * options.rho0 / 3.
                masses.append(Mm)
                ax_m_r.scatter(r.r, Mm)
                ax_m_r.annotate(k, [r.r, Mm])

                ax_u_r.scatter(r.r, v, c='g')
                ax_u_r.annotate(k, [r.r, v])
            except ValueError as e:
                ax_u_r.scatter(r.r, v, c='r')

        plt.show()
        print("Mass of M31: %s +- %s" % (
            np.format_float_scientific(np.mean(masses)),
            np.format_float_scientific(np.std(masses))
        ))

    def test_lv_unverial_model_fit(self):
        gals = load_leda(['../data/lv.dat'])
        invalid_gals = ['MESSIER031', 'Milky Way', 'HIZOA J1353-58', 'ESO006-001', 'dw1322-39']
        for g in invalid_gals:
            gals.pop(g)
        model = LinearFieldBiased()

        res_v, res_r, p, errors = model.fit_model(tuple(gals.values()), [0, 0, 0, 1, 1, 1, 60, 60])

        print(f"R0:    {res_r} +- {errors[:3]}")
        print(f"V0:    {res_v} +- {errors[3:6]}")
        print(f"a, b:  {p[0]}, {p[1]} +- {errors[6:]}")

    def test_lv_unverial_point_model(self):
        gals = load_leda(['../data/lv.dat'])
        invalid_gals = ['MESSIER031', 'Milky Way', 'HIZOA J1353-58', 'ESO006-001', 'dw1322-39']
        for g in invalid_gals:
            gals.pop(g)
        model = DenseDistField('point')

        res_v, res_r, p, errors = model.fit_model(tuple(gals.values()))

        print(f"R0:    {res_r} +- {errors[:3]}")
        print(f"V0:    {res_v} +- {errors[3:6]}")
        print(f"L0:    {p[0]} +- {errors[6:]}")

    def test_lv_unverial_point_model_fixed(self):
        gals = load_leda(['../data/lv.dat'])
        invalid_gals = ['MESSIER031', 'Milky Way', 'HIZOA J1353-58', 'ESO006-001', 'dw1322-39']
        mw = gals['Milky Way'].coordinates
        for g in invalid_gals:
            gals.pop(g)
        model = DenseDistField('point')

        res_v, p, errors = model.fit_model_fixed_pos(tuple(gals.values()), r0=mw)

        print(f"V0:    {res_v} +- {errors[:3]}")
        print(f"L0:    {p[0]} +- {errors[3:]}")
        print(f"V0 galactic {res_v.coords_icrs(u.km / u.s).galactic}")

    def test_mass_mw_m31(self):
        gals = load_leda(['../data/lv.dat'])
        model = DenseDistMassMwM31(gals['MESSIER031'].coordinates, gals['Milky Way'].coordinates, 'point')
        invalid_gals = ['MESSIER031', 'Milky Way', 'HIZOA J1353-58', 'ESO006-001', 'dw1322-39']
        for g in invalid_gals:
            gals.pop(g)

        res_v, p, errors = model.fit_model_fixed_pos(tuple(gals.values()), model.mw, [0, 0, 0, 1, 0.5])

        print(f"V0:    {res_v} +- {errors[:3]}")
        print(f"L0, w:    {p} +- {errors[3:]}")
        print(f"V0 galactic {res_v.coords_icrs(u.km / u.s).galactic}")

    def test_two_point_field(self):
        gals = load_leda(['../data/lv.dat'])
        gals = dict((k, v) for k, v in gals.items() if v.coordinates.r < 1.5)
        mw = gals['Milky Way']
        andromeda = gals['MESSIER031']
        gals.pop('Milky Way')
        gals.pop('MESSIER031')
        opts = PhysicOptions(omega_m_0=0.3, H0=72.0)

        model = TwoMassDenseDistField(opts, mw.coordinates, andromeda.coordinates, 'point', 'point')
        res, err = model.fit_model_fixed_apex(gals.values(), opts.apex, [0.2, 0.5])
        print(res)
        print(err)
