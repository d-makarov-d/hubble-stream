from __future__ import annotations
import numpy as np
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
from typing import Iterable, Sequence, Any
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
