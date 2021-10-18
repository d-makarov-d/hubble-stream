"""A solver for equation 10 and 14 from [https://ui.adsabs.harvard.edu/abs/2019MNRAS.490L..38B/abstract]"""
from __future__ import annotations

from typing import Union, Callable
import numpy as np
import scipy.integrate as integrate
from scipy import optimize
from astropy.cosmology import WMAP9 as cosmo
# TODO stick to astropy Quantity
# The solver operates equally on numbers and arrays
Data = Union[np.ndarray, float]


def _apply_to_data(data: Data, func: Callable[[float], float]) -> Data:
    if isinstance(data, np.ndarray):
        return np.vectorize(func)(data)
    else:
        return func(data)


class Solver:
    def __init__(self, options: SolverOptions):
        self.options = options
        omega = self.options.omega_m_0
        self.const_part = 2 / 3 * np.arccosh(omega ** -0.5)

    def dep(self, h: float, a: float) -> Data:
        """
        Describes the dependency, from equation 10
        :param h: Variable
        :param a: Variable
        :return: Function calculation result
        TODO compute U using h
        """
        O = self.options.omega_m_0
        H0 = self.options.H0
        U = (h / H0) ** 2 / O
        if U == 0:
            integ_func = lambda x: 2 * np.sqrt((1 - x ** 2) / (a - (1 - x ** 2) * (2 - x ** 2)))
        else:
            integ_func = lambda x: np.sqrt(x / (x ** 3 + (U - a - 1) * x + a))

        if h >= 0:
            # equation [10]
            integ_part, _ = integrate.quad(integ_func, 0, 1)
        else:
            # equation [14]
            im_sq_3 = -1j*np.sqrt(3)
            sq_3 = (-a - np.sqrt(27 * a ** 2 + 4 * (U - a - 1) ** 3)) ** (1 / 3)
            x_max = (1 - im_sq_3) * (U - a - 1) / (3 * 2 ** (3/4) * sq_3) - (1 + im_sq_3) * sq_3 / (2 ** (4 / 3))
            if abs(np.imag(x_max)) > 0:
                # x_max must have no complex part
                # TODO: use logger
                print('WARNING, could not correctly compute x_max, complex part not 0')
            x_max = np.real(x_max)

            integ_part1, _ = integrate.quad(integ_func, 0, x_max)
            integ_part2, _ = integrate.quad(integ_func, 0, 1)
            integ_part = 2 * integ_part1 - integ_part1

        return integ_part - self.const_part

    def _h(self, a: float) -> float:
        return optimize.newton_krylov(lambda h: self.dep(h, a), 0)

    def _a(self, h: float) -> float:
        return optimize.brentq(lambda a: self.dep(h, a), 2 + 1e-10, 15)

    def h(self, a: Data) -> Data:
        """
        Solution for dep(h, a=a) == 0
        :param a: values for which the equation needs to be solved
        :return: Solution, same shape as a
        """
        return _apply_to_data(a, self._h)

    def a(self, h: Data) -> Data:
        """
        Solution for dep(h=h, a) == 0
        :param h: values for which the equation needs to be solved
        :return: Solution, same shape as U
        """
        return _apply_to_data(h, self._a)


class SolverOptions:
    """Holds options for the solver"""
    def __init__(self, omega_m_0=0.0, H0=cosmo.H(0)):
        self._omega_m_0 = omega_m_0
        self._H0 = H0

    def set_omega_m_0(self, val: float):
        self._omega_m_0 = val
        return self

    @property
    def omega_m_0(self) -> float:
        return self._omega_m_0

    @property
    def H0(self) -> float:
        return self._H0.value
