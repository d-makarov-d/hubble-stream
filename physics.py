"""Some classes to describe physics and system properties"""
# TODO stick to astropy Quantity
from typing import Callable
from abc import ABC, abstractmethod
from astropy.cosmology import WMAP9 as cosmo
from astropy.units import Quantity

from eq_10 import Solver


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
