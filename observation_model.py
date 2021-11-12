from abc import ABC, abstractmethod
import numpy as np
from typing import Iterable
from scipy import optimize
import multiprocessing as mp
from joblib import Parallel, delayed

from util import Vector


class Galaxy:
    def __init__(self, coordinates: Vector, velocity: float):
        self.coordinates = coordinates
        self.velocity = velocity


class VelocityField(ABC):
    """Describes a velocity field, induced by some mass distribution"""
    def velocity(self, r0: Vector, v0: Vector, r: Vector) -> float:
        """
        Modeled observed velocity for this velocity field model
        :param r0: Observer position
        :param v0: Observer speed
        :param r: Vector from observer to object
        :return: Observed velocity
        """
        u = self.field(r0 + r)
        ni = Vector.unit(r)
        v0i = ni.dot(v0)
        vi = ni.dot(u)
        return vi - v0i

    @staticmethod
    def observer_velocity(galaxies: Iterable[Galaxy]) -> Vector:
        """
        Finds observer coordinates, using number of observed velocities
        :param galaxies: Observed galaxies
        :return: Vector of observer motion
        """
        obs_vels = np.array(tuple(map(
            lambda el: el.velocity,
            galaxies
        )))
        coords = np.array(tuple(map(
            lambda el: el.coordinates.cart,
            galaxies
        )))
        # data = np.concatenate([coords, np.expand_dims(obs_vels, 1)], 1).shape
        r = np.repeat(np.expand_dims(
            np.sum(coords ** 2, axis=1) ** 0.5, 1
        ), coords.shape[1], axis=1)
        unit_vects = coords / r

        def f(v):
            a = obs_vels + np.dot(unit_vects, v)
            return a
        res = optimize.least_squares(f, [0, 0, 0])
        return Vector.get_cart(res.x)

    def fit_model(self, galaxies: Iterable[Galaxy]) -> tuple[Vector, Vector]:
        """Fits the model to observation data, finding observer velocity and position
        :param galaxies: Observation data
        :return
            res_v: Vector, Observer velocity
            res_r: Vector, Observer position
        """
        obs_vels = np.array(tuple(gal.velocity for gal in galaxies))
        n_jobs = mp.cpu_count() - 1
        if len(obs_vels) < n_jobs:
            n_jobs = 1
        par = Parallel(n_jobs)

        def f(v):
            vel = Vector.get_cart(v[:3])
            r = Vector.get_cart(v[3:])
            model_vels = np.array(par(
                delayed(lambda coord: self.velocity(r, vel, coord))(gal.coordinates)
                for gal in galaxies
            ))
            # model_vels = np.zeros_like(obs_vels)
            # for i, gal in enumerate(galaxies):
            #     model_vels[i] = self.velocity(r, vel, gal.coordinates)

            return obs_vels - model_vels

        res = optimize.least_squares(f, [0, 0, 0, 1, 1, 1])
        res_v = Vector.get_cart(res.x[:3])
        res_r = Vector.get_cart(res.x[3:])
        return res_v, res_r

    @abstractmethod
    def field(self, coord: Vector) -> Vector:
        """Returns expected velocity vector for given coordinates, relative to mass center"""
        pass


class MassCenterModel(VelocityField):
    """Velocity field model, using mass center between Milky Way and M31"""
    def __init__(self, Mw: Vector, M31: Vector):
        """
        :param Mw: Coord vector from zero (Solar system) to Milky way center
        :param M31: Coord vector from zero (Solar system) to M31
        """
        self.Mw = Mw
        self.M31 = M31

    def field(self, coord: Vector) -> Vector:
        pass
