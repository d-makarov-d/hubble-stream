from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
from scipy.optimize._lsq.common import \
    solve_lsq_trust_region, evaluate_quadratic, \
    update_tr_radius, check_termination, \
    make_strictly_feasible
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
from typing import Iterable, Sequence, Any, Callable
import matplotlib.pyplot as plt


def draw_vectors(coords: Sequence[Vector], vectors: Sequence[Vector], norm_len=None):
    dots = np.zeros((len(coords), 3))
    vels = np.zeros((len(coords) * 3, 3))
    for i, (coord, vect) in enumerate(zip(coords, vectors)):
        dots[i, :] = coord.cart
        vels[3 * i, :] = coord.cart
        vels[3 * i + 1, :] = vect.cart
        vels[3 * i + 2, :] = np.nan
    vects = vels[1::3, :]
    if norm_len is not None:
        vects /= max(np.sum(vects**2, axis=1)**0.5) / norm_len
    vects += vels[0::3, :]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    c = np.array(len(coords) // 2 * ['r'])
    c = np.concatenate([c, np.array(len(coords) // 2 * ['g'])])
    ax.scatter(dots[:, 0], dots[:, 1], dots[:, 2], s=2, c=c)
    ax.plot(vels[:, 0], vels[:, 1], vels[:, 2], 'r', alpha=0.3)
    plt.show()


def _cartesian_to_spherical(vect: np.ndarray) -> np.ndarray:
    return np.array(tuple(map(
        lambda el: el.value,
        cartesian_to_spherical(*vect)
    )), dtype=float)


def _spherical_to_cartesian(vect: np.ndarray) -> np.ndarray:
    return np.array(tuple(map(
        lambda el: el.value,
        spherical_to_cartesian(*vect)
    )), dtype=float)


def _check_array(arr: Any) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        return np.array(arr)
    return arr


def _check_float(val: Any) -> float:
    if not isinstance(val, float):
        return float(val)
    return val


def least_square_xy(
        fun: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        method='2-point'
) -> tuple[np.ndarray, np.ndarray]:
    """Least squares method, assuming x has an error too
    :param fun: Function to optimize
    :param x0: Start guess
    :param method: ['2-point', '3-point'] Method of computing Jacobian matrix
    :return
        res: Solution
        err: Error
    """
    ftol = 1e-8
    xtol = 1e-8
    gtol = 1e-8
    x0 = np.atleast_1d(x0).astype(float)

    def jac_wrapped(_x0: np.ndarray, _f0: np.ndarray = None):
        """
        Compute finite difference approximation of the derivatives of a vector-valued function.
        :param _x0: Point at which to estimate the derivatives.
        :param _f0: fun(x0)
        :return: Jacobian matrix
        """
        if _f0 is None:
            _f0 = fun(_x0)

        # compute absolute step from relative step
        # the default EPS value
        EPS = np.finfo(np.float64).eps
        if method == "2-point":
            rel_step = EPS ** 0.5
        elif method == "3-point":
            rel_step = EPS ** (1 / 3)
        else:
            raise AttributeError('Unknown method: %s' % method)

        sign_x0 = (x0 >= 0).astype(float) * 2 - 1
        h = rel_step * sign_x0 * np.maximum(1.0, np.abs(_x0))
        if method == "2-point":
            use_one_sided = np.ones_like(h, dtype=bool)
        elif method == "3-point":
            h = np.abs(h)
            use_one_sided = np.zeros_like(h, dtype=bool)

        # compute jacobian
        m = _f0.size
        n = _x0.size
        J_transposed = np.empty((n, m))
        h_vecs = np.diag(h)

        for i in range(h.size):
            if method == '2-point':
                x = _x0 + h_vecs[i]
                dx = x[i] - _x0[i]  # Recompute dx as exactly representable number.
                df = fun(x) - _f0
            elif method == '3-point' and use_one_sided[i]:
                x1 = _x0 + h_vecs[i]
                x2 = _x0 + 2 * h_vecs[i]
                dx = x2[i] - _x0[i]
                f1 = fun(x1)
                f2 = fun(x2)
                df = -3.0 * _f0 + 4 * f1 - f2
            elif method == '3-point' and not use_one_sided[i]:
                x1 = _x0 - h_vecs[i]
                x2 = _x0 + h_vecs[i]
                dx = x2[i] - x1[i]
                f1 = fun(x1)
                f2 = fun(x2)
                df = f2 - f1
            else:
                raise RuntimeError("Never be here.")

            J_transposed[i] = df / dx

        if m == 1:
            J_transposed = np.ravel(J_transposed)

        return J_transposed.T

    f0 = fun(x0)
    J0 = jac_wrapped(x0, f0)
    x_scale = 1.

    # TRF method routine
    x = x0.copy()

    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1
    m, n = J.shape

    cost = 0.5 * np.dot(f, f)
    g = J.T.dot(f)
    scale, scale_inv = x_scale, 1 / x_scale

    Delta = norm(x0 * scale_inv)
    if Delta == 0:
        Delta = 1.0

    max_nfev = x0.size * 100
    alpha = 0.0  # "Levenberg-Marquardt" parameter
    step_norm = None
    actual_reduction = None

    while True:
        g_norm = norm(g, ord=np.inf)
        if g_norm < gtol:
            break
        if nfev == max_nfev:
            break

        d = scale
        g_h = d * g

        J_h = J * d
        U, s, V = svd(J_h, full_matrices=False)
        V = V.T
        uf = U.T.dot(f)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            step_h, alpha, n_iter = solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=alpha)

            predicted_reduction = -evaluate_quadratic(J_h, g_h, step_h)
            step = d * step_h
            x_new = x + step
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            # Usual trust-region step quality estimation.
            cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new

            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, step_h_norm > 0.95 * Delta)

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac_wrapped(x, f)
            njev += 1

            g = J.T.dot(f)
        else:
            step_norm = 0
            actual_reduction = 0

    res = x
    cov = np.linalg.inv(J.T.dot(J))
    var = np.sqrt(np.diagonal(cov))
    err = var

    return res, err


class Vector:
    """3D vector, supporting basic math operations"""
    def __init__(self, cart: np.ndarray):
        self._cart = cart
        self._sph = _cartesian_to_spherical(cart)

    @staticmethod
    def get_cart(cart: Iterable[float]) -> Vector:
        return Vector(np.array(cart))

    @staticmethod
    def get_sph(sph: Iterable[float]) -> Vector:
        cart = _spherical_to_cartesian(np.array(sph))
        return Vector(cart)

    @staticmethod
    def unit(vec: Vector) -> Vector:
        """Returns unit vector"""
        return Vector.get_cart(vec.cart / abs(vec))

    @property
    def cart(self) -> np.ndarray:
        return self._cart

    @cart.setter
    def cart(self, cart: np.ndarray):
        cart = _check_array(cart)
        self._cart = cart
        self._sph = _cartesian_to_spherical(cart)

    @property
    def sph(self) -> np.ndarray:
        return self._sph

    @sph.setter
    def sph(self, sph: np.ndarray):
        sph = _check_array(sph)
        self._cart = _spherical_to_cartesian(sph)
        self._sph = sph

    @property
    def x(self) -> float:
        return self._cart[0]

    @x.setter
    def x(self, x: float):
        x = _check_float(x)
        self._cart[0] = x
        self.cart = self._cart

    @property
    def y(self) -> float:
        return self._cart[1]

    @y.setter
    def y(self, y: float):
        y = _check_float(y)
        self._cart[1] = y
        self.cart = self._cart

    @property
    def z(self) -> float:
        return self._cart[2]

    @z.setter
    def z(self, z: float):
        z = _check_float(z)
        self._cart[2] = z
        self.cart = self._cart

    @property
    def r(self) -> float:
        return self._sph[0]

    @r.setter
    def r(self, r: float):
        r = _check_float(r)
        self._sph[0] = r
        self.sph = self._sph

    @property
    def lat(self) -> float:
        return self._sph[1]

    @lat.setter
    def lat(self, lat: float):
        lat = _check_float(lat)
        self._sph[1] = lat
        self.sph = self._sph

    @property
    def lon(self) -> float:
        return self._sph[2]

    @lon.setter
    def lon(self, lon: float):
        lon = _check_float(lon)
        self._sph[2] = lon
        self.sph = self._sph

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector.get_cart(self.cart + other.cart)
        return Vector.get_cart(self.cart + other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector.get_cart(self.cart - other.cart)
        return Vector.get_cart(self.cart - other)

    def __mul__(self, other):
        # Cross product as the default
        if isinstance(other, Vector):
            return self.cross(other)
        return Vector.get_cart(self.cart * other)

    def __truediv__(self, other):
        if isinstance(other, Vector):
            NotImplemented('Vector by vector division not implemented')
        return Vector.get_cart(self.cart / other)

    def __abs__(self):
        return self.r

    def __str__(self):
        return '[%f, %f, %f>' % (self.x, self.y, self.z)

    def __repr__(self):
        return self.__str__()

    def cross(self, other: Vector) -> Vector:
        cart = np.cross(self.cart, other.cart)
        return Vector.get_cart(cart)

    def dot(self, other: Vector) -> float:
        return np.dot(self.cart, other.cart)
