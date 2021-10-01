"""A solver for equation 10 and 14 from [https://ui.adsabs.harvard.edu/abs/2019MNRAS.490L..38B/abstract]"""
from __future__ import annotations

from typing import Union, Callable
import numpy as np
import scipy.integrate as integrate
from scipy import optimize
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

    def dep(self, U: float, a: float) -> Data:
        """
        Describes the dependency, from equation 10
        :param U: Variable
        :param a: Variable
        :return: Function calculation result
        """
        if U >= 0:
            if U == 0:
                integ_func = lambda x: 2 * np.sqrt((1 - x ** 2) / (a - (1 - x ** 2) * (2 - x ** 2)))
            else:
                integ_func = lambda x: np.sqrt(x / (x ** 3 + (U - a - 1) * x + a))
            integ_part, _ = integrate.quad(integ_func, 0, 1)
        else:
            raise NotImplementedError

        return integ_part - self.const_part

    def _U(self, a: float) -> float:
        return optimize.newton_krylov(lambda U: self.dep(U, a), 0)

    def _a(self, U: float) -> float:
        return optimize.brentq(lambda a: self.dep(U, a), 2 + 1e-10, 15)

    def U(self, a: Data) -> Data:
        """
        Solution for dep(U, a=a) == 0
        :param a: values for which the equation needs to be solved
        :return: Solution, same shape as a
        """
        return _apply_to_data(a, self._U)

    def a(self, U: Data) -> Data:
        """
        Solution for dep(U=U, a) == 0
        :param U: values for which the equation needs to be solved
        :return: Solution, same shape as U
        """
        # TODO
        return _apply_to_data(U, self._a)


class SolverOptions:
    """Holds options for the solver"""
    def __init__(self, omega_m_0=0.0):
        self._omega_m_0 = omega_m_0

    def set_omega_m_0(self, val: float):
        self._omega_m_0 = val
        return self

    @property
    def omega_m_0(self) -> float:
        return self._omega_m_0
