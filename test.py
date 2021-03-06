import unittest

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import optimize
from astropy import units as u
from astropy import constants as const

from eq_10 import Solver
from observation_model import VelocityField, Galaxy
from physics import PhysicOptions, Point, Halo, Empty, DenseDistField, DenseDistMassMwM31, TwoMassDenseDistField
from util import Vector, _cartesian_to_spherical, draw_vectors, least_square_xy, J
from files import load_vizier, load_leda


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
    def field(self, coord: Vector) -> Vector:
        return Vector.get_cart([0, 0, 0])


class LinearField(VelocityField):
    def field(self, coord: Vector, h0) -> Vector:
        return Vector.get_sph([h0 * (coord.r - 1), coord.lat, coord.lon])


class LinearFieldBiased(VelocityField):
    def field(self, coord: Vector, a, b) -> Vector:
        return Vector.get_sph([a * coord.r + b, coord.lat, coord.lon])


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

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        draw_vectors(ax, coords, vels, norm_len=30, s=2)
        plt.show()

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

    def test_lv_unverial_plot(self):
        gals = load_leda(['data/lv.dat'], vel='VLG')
        mw = gals['Milky Way']
        andromeda = gals['MESSIER031']
        center = mw.coordinates + (andromeda.coordinates - mw.coordinates) * 0.5
        gals = dict((k, Galaxy(v.coordinates - center, v.velocity)) for k, v in gals.items())

        dists = np.array([g.coordinates.r for g in gals.values()])
        vels = np.array([g.velocity for g in gals.values()])

        a, b = np.polyfit(dists, vels, 1)
        x = np.array([min(dists), max(dists)])
        _, ax = plt.subplots(1, 2)
        ax[0].scatter(dists, vels)
        ax[0].plot(x, x * a + b, 'r')
        ax[0].set_title('Velocities')
        ax[0].set_xlabel('distance')
        ax[0].set_ylabel('velocity')
        for i, txt in enumerate(gals.keys()):
            ax[0].annotate(txt, (dists[i] - 0.03 * len(txt) / 2, vels[i] + 4))
        pec_v = vels - (dists * a + b)
        ax[1].hist(pec_v)
        ax[1].set_title('Pecular velocities')
        plt.show()


    def test_lv_unverial_mw_m31_zones_plot(self):
        gals = load_leda(['data/lv.dat'], vel='VLG')
        gals = dict((k, v) for k, v in gals.items() if v.coordinates.r < 1.5)
        mw = gals['Milky Way']
        andromeda = gals['MESSIER031']
        gals.pop('Milky Way')
        gals.pop('MESSIER031')

        coords = [g.coordinates for g in gals.values()]
        vels = [Vector.get_sph([g.velocity, g.coordinates.lat, g.coordinates.lon]) for g in gals.values()]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        draw_vectors(ax, coords, vels, norm_len=2, s=5)
        ax.scatter(*mw.coordinates.cart, s=10, c='r')
        ax.text(*mw.coordinates.cart, 'Milky Way')
        ax.scatter(*andromeda.coordinates.cart, s=10, c='r')
        ax.text(*andromeda.coordinates.cart, 'M31')

        r_mw = []
        r_m31 = []
        c = []
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('Dist from MW')
        ax.set_ylabel('Dist from M31')
        for name, gal in gals.items():
            r_mw.append((gal.coordinates - mw.coordinates).r)
            r_m31.append((gal.coordinates - andromeda.coordinates).r)
            c.append(gal.velocity)
            ax.annotate(name, [r_mw[-1], r_m31[-1]])
        sc = ax.scatter(r_mw, r_m31, c=c, cmap=plt.get_cmap('cool'))
        plt.colorbar(sc)

        plt.show()

    def test_close_to_mw(self):
        gals = load_leda(['data/lv_all.dat'], vel='Vh')
        gals = dict((k, v) for k, v in gals.items() if v.coordinates.r < 1.5)
        mw = gals['Milky Way']
        andromeda = gals['MESSIER031']
        gals.pop('Milky Way')
        gals.pop('MESSIER031')
        v_apex = PhysicOptions().apex
        for gal in gals.values():
            gal.velocity = gal.velocity + v_apex.dot(Vector.unit(gal.coordinates))

        r_mw = []
        r_m31 = []
        c = []
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('Dist from MW')
        ax.set_ylabel('Dist from M31')
        for name, gal in gals.items():
            r_mw.append((gal.coordinates - mw.coordinates).r)
            r_m31.append((gal.coordinates - andromeda.coordinates).r)
            d = gal.coordinates
            r = d - mw.coordinates
            # v = (gal.velocity + v_apex.dot(d) / abs(d)) * (abs(r) * abs(d)) / (r.dot(d))
            v = abs(d - mw.coordinates) * abs(d) / (abs(d) ** 2 - mw.coordinates.dot(d)) * gal.velocity
            c.append(v)
            ax.annotate(name, [r_mw[-1], r_m31[-1]])
        sc = ax.scatter(r_mw, r_m31, c=c, cmap=plt.get_cmap('cool'))
        plt.colorbar(sc)
        plt.show()

    def test_mass_mw(self):
        gals = load_leda(['data/lv_all.dat'], vel='Vh')
        mw = gals['Milky Way']
        andromeda = gals['MESSIER031']
        gals.pop('Milky Way')
        gals.pop('MESSIER031')
        options = PhysicOptions(0.3, H0=73)
        v_apex = options.apex
        for gal in gals.values():
            gal.velocity = gal.velocity + v_apex.dot(Vector.unit(gal.coordinates))

        selected_gals = [gals['Eridanus 2'], gals['LeoT'], gals['Phoenix']]
        """selected_gals = list(
            v for v in gals.values()
            if 0.2 < (v.coordinates - mw.coordinates).r < 0.47 and (v.coordinates - mw.coordinates).r < (v.coordinates - andromeda.coordinates).r
        )"""
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
        gals = load_leda(['data/lv_all.dat'], vel='Vh')
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
        gals = load_leda(['data/lv.dat'])
        invalid_gals = ['MESSIER031', 'Milky Way', 'HIZOA J1353-58', 'ESO006-001', 'dw1322-39']
        for g in invalid_gals:
            gals.pop(g)
        model = LinearFieldBiased()

        res_v, res_r, p, errors = model.fit_model(tuple(gals.values()), [0, 0, 0, 1, 1, 1, 60, 60])

        print(f"R0:    {res_r} +- {errors[:3]}")
        print(f"V0:    {res_v} +- {errors[3:6]}")
        print(f"a, b:  {p[0]}, {p[1]} +- {errors[6:]}")

    def test_lv_unverial_point_model(self):
        gals = load_leda(['data/lv.dat'])
        invalid_gals = ['MESSIER031', 'Milky Way', 'HIZOA J1353-58', 'ESO006-001', 'dw1322-39']
        for g in invalid_gals:
            gals.pop(g)
        model = DenseDistField('point')

        res_v, res_r, p, errors = model.fit_model(tuple(gals.values()))

        print(f"R0:    {res_r} +- {errors[:3]}")
        print(f"V0:    {res_v} +- {errors[3:6]}")
        print(f"L0:    {p[0]} +- {errors[6:]}")

    def test_lv_unverial_point_model_fixed(self):
        gals = load_leda(['data/lv.dat'])
        invalid_gals = ['MESSIER031', 'Milky Way', 'HIZOA J1353-58', 'ESO006-001', 'dw1322-39']
        mw = gals['Milky Way'].coordinates
        for g in invalid_gals:
            gals.pop(g)
        model = DenseDistField('point')

        res_v, p, errors = model.fit_model_fixed_pos(tuple(gals.values()), r0=mw)

        print(f"V0:    {res_v} +- {errors[:3]}")
        print(f"L0:    {p[0]} +- {errors[3:]}")
        print(f"V0 galactic {res_v.coords_icrs(u.km / u.s).galactic}")

    def test_reference(self):
        """Compare new implementation to the reference implementation values"""
        gals = load_leda(['data/reference.dat'], ra='l', dec='b', dist='Dist', vel='Vh').values()
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

    def test_mass_mw_m31(self):
        gals = load_leda(['data/lv.dat'])
        model = DenseDistMassMwM31(gals['MESSIER031'].coordinates, gals['Milky Way'].coordinates, 'point')
        invalid_gals = ['MESSIER031', 'Milky Way', 'HIZOA J1353-58', 'ESO006-001', 'dw1322-39']
        for g in invalid_gals:
            gals.pop(g)

        res_v, p, errors = model.fit_model_fixed_pos(tuple(gals.values()), model.mw, [0, 0, 0, 1, 0.5])

        print(f"V0:    {res_v} +- {errors[:3]}")
        print(f"L0, w:    {p} +- {errors[3:]}")
        print(f"V0 galactic {res_v.coords_icrs(u.km / u.s).galactic}")

    def test_two_point_field(self):
        gals = load_leda(['data/lv.dat'])
        gals = dict((k, v) for k, v in gals.items() if v.coordinates.r < 1.5)
        mw = gals['Milky Way']
        andromeda = gals['MESSIER031']
        gals.pop('Milky Way')
        gals.pop('MESSIER031')
        opts = PhysicOptions(omega_m_0=0.3, H0=72.0)

        model = TwoMassDenseDistField(opts, mw.coordinates, andromeda.coordinates, 'point', 'point')
        res, err = model.fit_model_fixed_apex(gals.values(), opts.apex)
        print(res)
        print(err)
