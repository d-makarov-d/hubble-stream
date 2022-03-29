import unittest

import matplotlib.pyplot as plt
from util import Vector, draw_vectors
from files import load_vizier


class TestFiles(unittest.TestCase):
    def test_load_vizier(self):
        gals = load_vizier(['../data/table2.dat'], ['../data/vel1.dat'])
        coords = [gal.coordinates for gal in gals.values()]
        vels = [gal.velocity for gal in gals.values()]
        vels = [Vector.get_sph([v, c.lat, c.lon]) for v, c in zip(vels, coords)]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        draw_vectors(ax, coords, vels, norm_len=30, s=2)
        plt.show()

    def _test_save_matlab(self):
        gals = load_vizier(['../data/table2.dat'], ['../data/vel1.dat'])
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
        savemat('../data/galaxies.mat', data)