"""Some classes to describe physics and system properties"""
# TODO stick to astropy Quantity
from typing import Callable, Union
from abc import ABC, abstractmethod
from astropy.cosmology import WMAP9 as cosmo
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
import math

from eq_10 import Solver
from observation_model import VelocityField
from util import Vector


class PhysicOptions:
    """Holds options physic constants"""
    def __init__(self, omega_m_0=0.0, H0=cosmo.H(0)):
        self._omega_m_0 = omega_m_0
        if isinstance(H0, Quantity):
            self._H0 = H0
        else:
            self._H0 = Quantity(H0)

    def set_omega_m_0(self, val: float):
        self._omega_m_0 = val
        return self

    @property
    def omega_m_0(self) -> float:
        return self._omega_m_0

    @property
    def omega_l_0(self) -> float:
        return 1 - self._omega_m_0

    @property
    def H0(self) -> float:
        return self._H0.value

    @property
    def rho0(self):
        h_si = (self.H0*(u.km / u.Mpc / u.s)).to(1/u.s)
        rho_si = 3 * h_si ** 2 / (8 * math.pi * const.G)
        return rho_si.to(u.M_sun / u.Mpc ** 3).value

    @property
    def apex(self) -> Vector:
        return Vector.get_cart([115.9267, -120.4386, 183.6051])


class DensDistr(ABC):
    def __init__(self, options: PhysicOptions):
        self._options = options
        self.sigma = Solver(options).a0 * options.omega_l_0

    @abstractmethod
    def model(self, L0: float) -> Callable[[float], float]:
        """Matter distribution model. Returns mass from radius dependency."""
        pass


class Point(DensDistr):
    def model(self, L0: float) -> Callable[[float], float]:
        omega = self._options.omega_m_0
        return lambda r0: (self.sigma - omega) * (L0 / r0) ** 3 + omega


class Halo(DensDistr):
    def model(self, L0: float) -> Callable[[float], float]:
        omega = self._options.omega_m_0
        return lambda r0: (self.sigma - omega) * (L0 / r0) ** 2 + omega


class Empty(DensDistr):
    def model(self, L0: float) -> Callable[[float], float]:
        return lambda r0: self.sigma * (L0 / r0) ** 3


class DenseDistField(VelocityField):
    def __init__(self, distribution: Union[str, DensDistr]):
        name_to_class = {
            'halo': Halo,
            'point': Point,
            'empty': Empty
        }
        if isinstance(distribution, str):
            cls = name_to_class.get(distribution)
            if cls is None:
                raise ValueError('distribution must be in ["halo", "point", "empty"]')
            opts = PhysicOptions(0.3)
            self.distribution = cls(opts)
            self.solver = Solver(opts)
            self.options = opts
        else:
            self.distribution = distribution
            self.solver = Solver(distribution._options)
            self.options = distribution._options

    def field(self, coord: Vector, L0) -> Vector:
        profile = self.distribution.model(L0)
        sigma = profile(coord.r)
        alpha = sigma / self.options.omega_l_0
        h = self.solver.h(alpha)
        v = h * coord.r
        return Vector.get_sph([v, coord.lat, coord.lon])


class TwoMassDenseDistField(VelocityField):
    def __init__(self, opts: PhysicOptions, mw: Vector, m31: Vector, distr_mw: str, dist_m31: str):
        self.solver = Solver(opts)
        self.options = opts
        self.mw = mw
        self.m31 = m31
        self.distr_mw = self._get_distr_cls(distr_mw, opts)
        self.distr_m31 = self._get_distr_cls(dist_m31, opts)

    def _get_distr_cls(self, dist_name: str, opts: PhysicOptions) -> DensDistr:
        name_to_class = {
            'halo': Halo(opts),
            'point': Point(opts),
            'empty': Empty(opts)
        }
        cls = name_to_class.get(dist_name)
        if cls is None:
            raise ValueError(f'distribution must be in ["halo", "point", "empty"], got "{dist_name}"')

        return cls

    def _calc_speed(self, sigma, coord):
        alpha = sigma / self.options.omega_l_0
        h = self.solver.h(alpha)
        v = h * coord.r
        return Vector.get_sph([v, coord.lat, coord.lon])

    def field(self, coord: Vector, LMW: float, LM31: float) -> Vector:
        sigma_m31 = self.distr_m31.model(LM31)(coord.r)
        sigma_mw = self.distr_mw.model(LMW)(coord.r)
        eps_lm31 = sigma_m31 * LM31 ** 3
        eps_lmw = sigma_mw * LMW ** 3
        gama =  eps_lm31  / (eps_lmw + eps_lm31)

        a = coord - self.mw
        b = coord - self.m31
        D = gama * (self.m31 - self.mw)
        r = coord - self.mw - D

        Umw = a * (self._calc_speed(sigma_mw, coord).r / a.r - self.options.H0)
        Um31 = b * (self._calc_speed(sigma_m31, coord).r / b.r - self.options.H0)

        Vcm = self.options.H0*r + Umw + Um31
        Vmw = D * (self.options.H0 * (1 - gama) - self._calc_speed(sigma_m31, D))

        return (Vcm - Vmw) * Vector.unit(coord)


class DenseDistMassMwM31(DenseDistField):
    """Dense distribution field with mass center between Milky Way and M31"""
    def __init__(self, mw: Vector, m31: Vector, distribution: Union[str, DensDistr]):
        """
        :param mw: Milky way coordinates
        :param m31: M31 coordinates
        :param distribution: Mass distribution type
        """
        self.mw_m31 = m31 - mw
        self.mw = mw
        super().__init__(distribution)

    def field(self, coord: Vector, L0, w) -> Vector:
        """
        :param coord: Coordinates
        :param L0: Model parameter
        :param w: Relative position on vector MW - M31. If w=0, mass center should be in MW, if w=1, in M31
        :return: Field vector in coord
        """
        center = self.mw + self.mw_m31 * w
        transformed = coord - center
        profile = self.distribution.model(L0)
        sigma = profile(transformed.r)
        alpha = sigma / self.options.omega_l_0
        h = self.solver.h(alpha)
        v = h * transformed.r
        return Vector.get_sph([v, transformed.lat, transformed.lon])
