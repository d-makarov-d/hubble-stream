import unittest

import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u

from files import load_leda
from observation_model import Galaxy, VelocityField
from physics import PhysicOptions
from util import Vector, draw_vectors, scatter_skymap, text_galactic, text_icrs

class Test(unittest.TestCase):
    def setUp(self) -> None:
        mw_coords = SkyCoord(l=0 * u.rad, b=0 * u.rad, distance=8.2 * u.kpc, frame='galactic').icrs
        self.mw_coords = Vector.get_sph([mw_coords.distance.value, mw_coords.dec.rad, mw_coords.ra.rad])
        self.f_in_dist = lambda gal: (gal.coordinates - self.mw_coords).r < 300

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

    def test_velocity_distribution(self):
        gals = load_leda(['../data/mw_satellites.dat'], ra='RAdeg', dec='DEdeg', dist='Dkpc', vel='Vh', name='Name')
        # apply apex
        options = PhysicOptions(0.3, H0=73)
        v_apex = options.apex
        for gal in gals.values():
            gal.velocity = gal.velocity + v_apex.dot(Vector.unit(gal.coordinates))
        # gals = dict((k, v) for k, v in gals.items() if (v.coordinates - self.mw_coords).r < 500)
        x = [(g.coordinates - self.mw_coords).r for g in gals.values()]
        y = [g.velocity for g in gals.values()]
        plt.scatter(x, y)
        for name, g in gals.items():
            r = (g.coordinates - self.mw_coords).r
            if r < 300 or r > 450:
                continue
            plt.scatter(r, g.velocity, c='r')
            plt.text(r, g.velocity, name)
        plt.xlabel('Distance from MW, kpc')
        plt.ylabel('Velocity, kpc/sec')
        plt.show()

    def test_plot_galaxies_distances_galactic(self):
        gals = load_leda(['../data/mw_satellites.dat'], ra='RAdeg', dec='DEdeg', dist='Dkpc', vel='Vh', name='Name')
        gals = dict((k, v) for k, v in gals.items() if np.log10((v.coordinates - self.mw_coords).r) < 2.5)
        d_mw = [(g.coordinates - self.mw_coords).r for g in gals.values()]
        d_mw = np.log10(d_mw)
        sc = scatter_skymap([v.coordinates for v in gals.values()], coords='galactic', c=d_mw, cmap='jet')
        plt.colorbar(sc, orientation='horizontal')
        text_galactic(plt.gca(), 'SMC', gals['SMC'].coordinates)
        text_galactic(plt.gca(), 'LMC', gals['LMC'].coordinates)
        plt.title('Milky way satellites', y=1.1)
        plt.show()

    def test_plot_galaxies_distances_icrs(self):
        gals = load_leda(['../data/mw_satellites.dat'], ra='RAdeg', dec='DEdeg', dist='Dkpc', vel='Vh', name='Name')
        gals = dict((k, v) for k, v in gals.items() if np.log10((v.coordinates - self.mw_coords).r) < 2.5)
        d_mw = [(g.coordinates - self.mw_coords).r for g in gals.values()]
        d_mw = np.log10(d_mw)
        sc = scatter_skymap([v.coordinates for v in gals.values()], coords='icrs', c=d_mw, cmap='jet')
        plt.colorbar(sc, orientation='horizontal')
        text_icrs(plt.gca(), 'SMC', gals['SMC'].coordinates)
        text_icrs(plt.gca(), 'LMC', gals['LMC'].coordinates)
        plt.title('Milky way satellites', y=1.1)
        plt.show()

    def test_cumulative_distance_distribution(self):
        gals = load_leda(['../data/mw_satellites.dat'], ra='RAdeg', dec='DEdeg', dist='Dkpc', vel='Vh', name='Name')
        gals_by_dist = sorted(gals.items(), key=lambda el: (el[1].coordinates - self.mw_coords).r)
        #gals_by_dist = ((k, v) for k, v in gals_by_dist if self.f_in_dist(v))
        gals_by_dist = dict(gals_by_dist)
        names = tuple(gals_by_dist.keys())
        x = [(gal.coordinates - self.mw_coords).r for gal in gals_by_dist.values()]
        y = np.arange(len(gals_by_dist))
        X = np.zeros(len(x) * 2 - 1)
        Y = np.zeros(len(y) * 2 - 1)
        for i in range(len(x) - 1):
            X[2 * i] = X[2*i +1] = x[i]
            Y[2 * i] = y[i]
            Y[2 * i + 1] = y[i + 1]
        X[-1] = x[-1]
        Y[-1] = y[-1]
        plt.plot(X, Y)
        plt.xscale('log')
        for i in range(len(x)):
            if 300 < x[i] < 450:
                plt.plot([X[2*i - 1], X[2*i]], [Y[2*i - 1], Y[2*i]], 'r')
                plt.text(X[2*i - 1], y[i] + 0.4, names[i], fontdict={'size': 8})
        plt.show()

    def test_radial_velociry_distribution(self):
        gals = load_leda(['../data/mw_satellites.dat'], ra='RAdeg', dec='DEdeg', dist='Dkpc', vel='Vh', name='Name')
        vels = np.array([gal.velocity for gal in gals.values()])
        # vels[np.isnan(vels)] = 0
        sc = scatter_skymap([v.coordinates for v in gals.values()], ax=ax, coords='galactic', c=vels, cmap='jet')
        plt.colorbar(sc, orientation='horizontal')
        plt.show()

    def test_running_apex(self):
        gals = load_leda(['../data/mw_satellites.dat'], ra='RAdeg', dec='DEdeg', dist='Dkpc', vel='Vh', name='Name')
        gals_by_dist = sorted(gals.items(), key=lambda el: (el[1].coordinates - self.mw_coords).r)
        gals_by_dist = dict((k, v) for k, v in gals_by_dist if not np.isnan(v.velocity) and self.f_in_dist(v) )
        d_mw = [(g.coordinates - self.mw_coords).r for g in gals_by_dist.values()]

        n0 = 5
        v = []
        for i in range(n0, len(gals_by_dist)):
            apex, err = VelocityField.observer_velocity(list(gals_by_dist.values())[:i])
            v.append(apex.r)

        plt.plot(d_mw[n0:], v, '.')
        plt.show()
