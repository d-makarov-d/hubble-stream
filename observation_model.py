from abc import ABC, abstractmethod
import numpy as np
from typing import Iterable, Sequence
from scipy import optimize
import multiprocessing as mp
from joblib import Parallel, delayed
from inspect import signature

from util import Vector


class Galaxy:
    def __init__(self, coordinates: Vector, velocity: float):
        self.coordinates = coordinates
        self.velocity = velocity


class VelocityField(ABC):
    """Describes a velocity field, induced by some mass distribution"""
    def velocity(self, r0: Vector, v0: Vector, r: Vector, *params) -> float:
        """
        Modeled observed velocity for this velocity field model
        :param r0: Observer position
        :param v0: Observer speed
        :param r: Vector from observer to object
        :param params: Parameters for field model
        :return: Observed velocity
        """
        u = self.field(r0 + r, *params)
        ni = Vector.unit(r)
        v0i = ni.dot(v0)
        vi = ni.dot(u)
        return vi - v0i

    @staticmethod
    def observer_velocity(galaxies: Iterable[Galaxy]) -> tuple[Vector, Vector]:
        """
        Finds observer coordinates, using number of observed velocities
        :param galaxies: Observed galaxies
        :return: Vector of observer motion, and vector of errors
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

        cov = np.linalg.inv(res.jac.T.dot(res.jac))
        var = np.sqrt(np.diagonal(cov))
        sigma = (sum(res.fun ** 2) / len(res.fun)) ** 0.5
        return Vector.get_cart(res.x), Vector.get_cart(var * sigma)

    def fit_model(self, galaxies: Iterable[Galaxy], x0: np.ndarray = None) -> tuple[Vector, Vector, Sequence[float], np.ndarray]:
        """Fits the model to observation data, finding observer velocity and position
        :param galaxies: Observation data
        :param x0: Initial guess
        :return
            res_v: Vector, Observer velocity
            res_r: Vector, Observer position
            res_p: Sequence[float], Model params
            errors: np.ndarray, Errors [:3] for res_v, [3:6] for res_r, [6:] for other params
        """
        obs_vels = np.array(tuple(gal.velocity for gal in galaxies))
        n_jobs = mp.cpu_count() - 1
        if len(obs_vels) < n_jobs:
            n_jobs = 1
        par = Parallel(n_jobs)

        def f(v):
            vel = Vector.get_cart(v[:3])
            r = Vector.get_cart(v[3:6])
            p = v[6:]
            model_vels = np.array(par(
                delayed(lambda coord: self.velocity(r, vel, coord, *p))(gal.coordinates)
                for gal in galaxies
            ))
            # model_vels = np.zeros_like(obs_vels)
            # for i, gal in enumerate(galaxies):
            #     model_vels[i] = self.velocity(r, vel, gal.coordinates)

            return obs_vels - model_vels

        n_additional_params = len(signature(self.field).parameters) - 1
        if x0 is None:
            x0 = [0, 0, 0, 1, 1, 1] + n_additional_params * [1]
        res = optimize.least_squares(f, x0)
        res_v = Vector.get_cart(res.x[:3])
        res_r = Vector.get_cart(res.x[3:6])

        cov = np.linalg.inv(res.jac.T.dot(res.jac))
        var = np.sqrt(np.diagonal(cov))
        sigma = (sum(res.fun ** 2) / len(res.fun)) ** 0.5

        return res_v, res_r, res.x[6:], var * sigma

    def fit_model_fixed_pos(
            self, galaxies: Iterable[Galaxy],
            r0: Vector, x0: np.ndarray = None
    ) -> tuple[Vector, np.ndarray, np.ndarray]:
        """Fits the model to observation data, finding observer velocity
        :param galaxies: Observation data
        :param r0: Observer position
        :param x0: Initial guess
        :return
            res_v: Vector, Observer velocity
            res_p: np.ndarray, Model params
            errors: np.ndarray, Errors [:3] for res_v, [3:] for other params
        """
        obs_vels = np.array(tuple(gal.velocity for gal in galaxies))
        n_jobs = mp.cpu_count() - 1
        if len(obs_vels) < n_jobs:
            n_jobs = 1
        par = Parallel(n_jobs)

        def f(v):
            vel = Vector.get_cart(v[:3])
            p = v[3:]
            model_vels = np.array(par(
                delayed(lambda coord: self.velocity(r0, vel, coord, *p))(gal.coordinates)
                for gal in galaxies
            ))

            return obs_vels - model_vels

        n_additional_params = len(signature(self.field).parameters) - 1
        if x0 is None:
            x0 = [0, 0, 0] + n_additional_params * [1]
        res = optimize.least_squares(f, x0)
        res_v = Vector.get_cart(res.x[:3])

        cov = np.linalg.inv(res.jac.T.dot(res.jac))
        var = np.sqrt(np.diagonal(cov))
        sigma = (sum(res.fun ** 2) / len(res.fun)) ** 0.5

        return res_v, res.x[3:], var * sigma

    def fit_model_fixed_apex(self, galaxies: Sequence[Galaxy], apex: Vector, x0=None):
        """
        Fits model with predefined apex
        :param galaxies: Observation data
        :param apex: Apex
        :param x0: Optional, initial params
        :return:
            res: np.ndarray fitted params
            err: np.ndarray errors
        """
        obs_vels = np.zeros(len(galaxies))
        # apply apex
        for i, gal in enumerate(galaxies):
            obs_vels[i] = gal.velocity + apex.dot(Vector.unit(gal.coordinates))
        n_jobs = mp.cpu_count() - 1
        if len(obs_vels) < n_jobs:
            n_jobs = 1
        par = Parallel(n_jobs)

        def f(v):
            model_vels = np.array(par(
                delayed(lambda coord: self.field(coord, *v).dot(Vector.unit(coord)))(gal.coordinates)
                for gal in galaxies
            ))

            return obs_vels - model_vels

        n_additional_params = len(signature(self.field).parameters) - 1
        if x0 is None:
            x0 = n_additional_params * [1]

        res = optimize.least_squares(f, x0)

        cov = np.linalg.inv(res.jac.T.dot(res.jac))
        var = np.sqrt(np.diagonal(cov))
        sigma = (sum(res.fun ** 2) / len(res.fun)) ** 0.5

        return res, var*sigma

    @abstractmethod
    def field(self, coord: Vector, *params) -> Vector:
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

    def field(self, coord: Vector, *params) -> Vector:
        pass
