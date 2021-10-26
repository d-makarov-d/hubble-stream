"""A solver for equation 10 and 14 from [https://ui.adsabs.harvard.edu/abs/2019MNRAS.490L..38B/abstract]"""
from __future__ import annotations

from typing import Union, Callable
import numpy as np
import scipy.integrate as integrate
from scipy import optimize
from joblib import Parallel, delayed
import multiprocessing as mp

# TODO stick to astropy Quantity
# The solver operates equally on numbers and arrays
Data = Union[np.ndarray, float]


def _apply_to_data(data: Data, func: Callable[[float], float]) -> Data:
    if isinstance(data, np.ndarray):
        def f(arr):
            return np.vectorize(func)(arr)
        n_jobs = mp.cpu_count() - 1
        split_dim = np.argmax(data.shape)
        if data.shape[split_dim] >= n_jobs:
            parts = np.array_split(data, n_jobs, axis=split_dim)
            # with mp.Pool(n_jobs) as p:
                # res = p.map(f, parts)
            res = Parallel(n_jobs)(delayed(f)(arr) for arr in parts)
            return np.concatenate(res, axis=split_dim)
        return np.vectorize(func)(data)
    else:
        return func(data)


def _solve(p, q):
    return np.real(sorted(np.roots([1, 0, p, q])))[1]


class Solver:
    def __init__(self, options):
        """
        :param options: Physical system options
        :type options: PhysicOptions
        """
        self.options = options
        omega = self.options.omega_m_0
        self.const_part = 2 / 3 * np.arccosh(omega ** -0.5)
        self.eps = np.finfo(float).eps
        self.a0 = optimize.brentq(lambda a: self.dep(0, a), 2 + self.eps ** 0.5, 15)
        h2 = optimize.brentq(lambda h: self.dep(h, 2.), 100 * self.eps ** 0.5, self._U_2_h(5))
        self._U2 = self._h_2_U(h2)

    def _h_2_U(self, h: float) -> float:
        Ol = self.options.omega_l_0
        H0 = self.options.H0
        return (h / H0) ** 2 / Ol

    def _U_2_h(self, U: float) -> float:
        Ol = self.options.omega_l_0
        H0 = self.options.H0
        return H0 * (U * Ol) ** 0.5

    def dep(self, h: float, a: float) -> Data:
        """
        Describes the dependency, from equation 10
        :param h: Variable
        :param a: Variable
        :return: Function calculation result
        TODO compute U using h
        """
        U = self._h_2_U(h)
        if abs(U) < self.eps:
            integ_func = lambda x: 2 * np.sqrt((1 - x ** 2) / (a - (1 - x ** 2) * (2 - x ** 2)))
        else:
            integ_func = lambda x: np.sqrt(x / (x ** 3 + (U - a - 1) * x + a))

        if h >= 0:
            # equation [10]
            integ_part, _ = integrate.quad(integ_func, 0, 1)
        else:
            # equation [14]
            # x_max = x_max_precomputed(U, a)
            # x_max = _solve_cubic(U - a - 1, a)
            x_max = _solve(U - a - 1, a)

            integ_part1, _ = integrate.quad(integ_func, 0, x_max)
            integ_part2, _ = integrate.quad(integ_func, 0, 1)
            integ_part = 2 * integ_part1 - integ_part2

        return integ_part - self.const_part

    def _h(self, a: float) -> float:
        if abs(a) < self.eps:
            return 0.
        elif a > self.a0:
            # negative speed
            h_min = - self._U_2_h(a + 1 - 3 * (0.5 * a) ** (2 / 3)) * (1 - 3 * self.eps ** 0.5)
            return optimize.brentq(lambda h: self.dep(h, a), h_min, 0)
        elif a < self.a0:
            # positive
            if a > 2:
                u_min = 0
                u_max = self._U2 * (1 + self.eps ** 0.5)
            else:
                u_min = a + 1 - 3 * (0.5 * a) ** (2 / 3) * (1 - 2 * self.eps ** 0.5)
                if u_min < self._U2:
                    u_min = self._U2 * (1 - self.eps ** 0.5)
                u_max = 2 * (a + 1)
            h_min = self._U_2_h(u_min)
            h_max = self._U_2_h(u_max)
            return optimize.brentq(lambda h: self.dep(h, a), h_min, h_max)

    def _a(self, h: float) -> float:
        if abs(h) < self.eps:
            return self.a0
        elif h > 0:
            U = self._h_2_U(h)
            if U < self._U2:
                left_bound = 2 * (1 - self.eps ** 0.5)
                right_bound = self.a0
            else:
                x = sorted(np.real(np.roots([2, -3, 0, 1 - U])))[1]
                alpha = 2 * x ** 3
                left_bound = abs(alpha * (1 + self.eps ** 0.5))
                right_bound = 2 * (1 + self.eps ** 0.5)

            return optimize.brentq(lambda a: self.dep(h, a), left_bound, right_bound)
        else:
            U = self._h_2_U(h)
            x = sorted(np.real(np.roots([2, -3, 0, 1 - U])))[2]
            alpha = 2 * x ** 3
            if self.a0 > alpha:
                left_bound = self.a0 * (1 - self.eps ** 0.5)
            else:
                left_bound = alpha * (1 + self.eps ** 0.5)
            if U - 1 > left_bound:
                # probably always satisfied
                left_bound = U - 1
            # TODO somebody's function
            right_bound = self.a0 / (1 + np.exp(0.1 * np.log10(U))) * alpha
            return optimize.brentq(lambda a: self.dep(h, a), left_bound, right_bound)

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

