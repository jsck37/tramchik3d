# -*- coding: utf-8 -*-
"""Векторная математика 3D-движка (numpy)."""
import math
import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def normalize(v):
    """Нормализация вектора (список/ndarray) → ndarray(3,)."""
    v = np.asarray(v, dtype=np.float64)
    n = math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0])
    return v / n


def rot_y_matrix(yaw):
    """Матрица поворота вокруг оси Y: R @ (0,0,1) = (sin, 0, cos)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def wrap_angle(a):
    """Нормализация угла в диапазон (-π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def lerp_angle(a, b, t):
    """Плавный поворот угла a к углу b с коэффициентом t (0..1)."""
    return a + wrap_angle(b - a) * min(1.0, max(0.0, t))


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)
