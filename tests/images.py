import unittest

import numpy as np
import matplotlib.pyplot as plt

from files import load_leda
from observation_model import Galaxy
from physics import PhysicOptions
from util import Vector, draw_vectors

class Test(unittest.TestCase):
    def test_lv_unverial_plot(self):
        gals = load_leda(['../data/lv.dat'], vel='VLG')
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
        gals = load_leda(['../data/lv.dat'], vel='VLG')
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
        gals = load_leda(['../data/lv_all.dat'], vel='Vh')
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
