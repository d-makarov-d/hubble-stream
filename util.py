from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
from scipy.optimize._lsq.common import \
    solve_lsq_trust_region, evaluate_quadratic, \
    update_tr_radius, check_termination
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy as hp
from typing import Iterable, Sequence, Any, Callable, Union
import matplotlib.pyplot as plt
from vector import Vector as MVector


def draw_vectors(ax, coords: Sequence[Vector], vectors: Sequence[Vector], norm_len=None, **dot_params):
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
    ax.scatter(dots[:, 0], dots[:, 1], dots[:, 2], **dot_params)
    ax.plot(vels[:, 0], vels[:, 1], vels[:, 2], 'r', alpha=0.3)


def _apply_dust(ax, coords: str) -> np.ndarray:
    extinction_map = hp.read_map('../heavy_data/HFI_SkyMap_857_2048_R1.10_nominal.fits')
    proj = hp.projector.CartesianProj()
    nside = hp.npix2nside(len(extinction_map))

    def f_icrs(X, Y, Z):
        r, lat, lon = cartesian_to_spherical(X, Y, Z)
        frame = SkyCoord(l=0.75 * np.pi * u.rad - lon, b=lat, distance=r, frame='galactic').icrs
        x, y, z = spherical_to_cartesian(frame.distance, frame.dec.rad, np.pi * 1.75 - frame.ra.rad)
        return hp.vec2pix(nside, x.value, y.value, z.value)

    def f_galactic(X, Y, Z):
        r, lat, lon = cartesian_to_spherical(X, Y, Z)
        x, y, z = spherical_to_cartesian(r, lat, (lon - np.pi * u.rad).wrap_at('180d'))
        return hp.vec2pix(nside, x.value, y.value, z.value)

    type_to_f = {
        'galactic': f_galactic,
        'icrs': f_icrs
    }
    f = type_to_f.get(coords)
    if f is None:
        raise ValueError('Unknown coordinate system "%s". supported are: %s' % (coords, list(type_to_f.keys())))

    im = proj.projmap(extinction_map, vec2pix_func=f)
    im[im < 7] = np.nan
    im = np.log(im - 5)
    x = np.linspace(-np.pi, np.pi, im.shape[1])
    y = np.linspace(-np.pi / 2, np.pi / 2, im.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.pcolormesh(X, Y, im, cmap='Greys')


def scatter_skymap(points: Sequence[Vector], coords: str, ax = None, **plt_params):
    """Draw points on skymap of dust extinction using aitoff projection"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='aitoff')

    _apply_dust(ax, coords)

    dots = np.zeros((len(points), 2))

    if coords == 'icrs':
        for i in range(0, len(points)):
            gal = points[i].coords_icrs(u.kpc)
            dots[i, :] = ((180 * u.deg - gal.ra).wrap_at('180d').rad, gal.dec.rad)
    elif coords == 'galactic':
        for i in range(0, len(points)):
            gal = points[i].coords_icrs(u.kpc).galactic
            dots[i, :] = (np.pi - gal.l.rad, gal.b.rad)
    else:
        raise ValueError('Unsupported coordinate system')

    ax.grid(True)
    ax.set_xticks(
        np.linspace(-np.pi, np.pi, 13),
        ['$24^h$', '', '', '$18^h$', '', '', '$12^h$', '', '', '$6^h$', '', '', '$0^h$']
    )
    ax.tick_params(direction='out')
    sc = ax.scatter(dots[:, 0], dots[:, 1], **plt_params)
    return sc

def text_galactic(ax, text: str, pos: Vector, **params):
    gal = pos.coords_icrs(u.kpc).galactic
    ax.text(np.pi - gal.l.rad, gal.b.rad, text)

def text_icrs(ax, text: str, pos: Vector, **params):
    gal = pos.coords_icrs(u.kpc)
    ax.text((180 * u.deg - gal.ra).wrap_at('180d').rad, gal.dec.rad, text)


def J(x0: np.ndarray, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Compute Jacobian (N, N) in point x0 (N, ), using function f"""
    N = x0.shape[0]
    eps = np.finfo(float).eps ** 0.5
    jac = np.empty((N, N))
    for i in range(N):
        dx = x0[i] * eps
        step = np.zeros_like(x0)
        step[i] = dx
        jac[:, i] = (f(x0 + step) - f(x0)) / dx

    return jac


def dec_vector_str(v: Vector, err: Vector) -> str:
    """
    Returns string representation of given vector with errors, in galactic coordinates
    :param v: Value, in cartesian coordinates
    :param err: Error for each value component, in cartesian coordinates
    :return: String representation in galactic coordinates, with error
    """
    x, y, z = v.cart
    r = v.r
    t1 = ((x**2 + y**2) / (x**2 + y**2 + z**2))**0.5
    t2 = (x**2 + y**2 + z**2)**1.5
    m = np.array([
        [x/r, y/r, z/r],                                # dr/dx, dr/dy, dr/dz
        [x*z / (t1 * t2), y*z / (t1 * t2), t1 / r],     # d(lat)/dx, d(lat)/dy, d(lat)/dz
        [- y / (x**2 + y**2), x / (x**2 + y**2), 0]     # d(lon)/dx, d(lon)/dy, d(lon)/dz
    ])
    s = np.array([err.cart]).T
    # galactic coordinates errors
    g_err = np.matmul(m, s)
    g_coord = v.coords_icrs(u.km / u.s).galactic
    return f"l: {g_coord.l.deg} +- {g_err[2] / np.pi * 180} deg" \
           f"b: {g_coord.b.deg} +- {g_err[1] / np.pi * 180} deg" \
           f"r: {g_coord.r.km} +- {g_err[0]} km"


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


class Vector(MVector):
    """3D vector, supporting basic math operations"""
    def __init__(self, cart: Union[Sequence, np.ndarray]):
        if isinstance(cart, np.ndarray):
            cart = tuple(cart)
        super().__init__(cart)

    @staticmethod
    def get_cart(cart: Iterable[float]) -> Vector:
        return Vector(cart)

    @staticmethod
    def get_sph(sph: Iterable[float]) -> Vector:
        cart = _spherical_to_cartesian(np.array(sph))
        return Vector(cart)

    def coords_icrs(self, units = u.Unit()):
        return SkyCoord(ra=self.lon*u.rad, dec=self.lat*u.rad, distance=self.r*units, frame='icrs')

    @staticmethod
    def unit(vec: Vector) -> Vector:
        """Returns unit vector"""
        return vec * (1 / vec.r)
