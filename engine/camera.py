# -*- coding: utf-8 -*-
"""Камера 3D-движка: орбитальная (облёт склада), следящая и фиксированная (бортовая)."""
import math
import numpy as np
from .vecmath import UP, normalize, clamp


class CamData:
    """Результат compute(): базис камеры + фокус + цель рендера."""
    __slots__ = ("pos", "u", "v", "w", "focal", "surface", "cx", "cy", "clip", "vw", "vh")

    def __init__(self, pos, u, v, w, focal):
        self.pos = pos
        self.u = u          # вправо
        self.v = v          # вверх
        self.w = w          # вперёд (от камеры)
        self.focal = focal
        self.surface = None
        self.cx = 0.0
        self.cy = 0.0
        self.clip = None
        self.vw = 0
        self.vh = 0


class Camera3D:
    def __init__(self, fov=60.0, mode="orbit"):
        self.fov = fov
        self.mode = mode                 # "orbit" | "fixed"
        self.target = np.array([0.0, 1.0, 0.0])
        self.dist = 30.0
        self.yaw = 0.75
        self.pitch = 0.46
        self.pos = np.array([0.0, 8.0, 30.0])
        self.fwd = np.array([0.0, -0.15, -1.0])
        self.min_dist, self.max_dist = 5.0, 70.0

    # --- управление орбитальной камерой ---
    def orbit(self, dx, dy):
        self.yaw += dx * 0.005
        self.pitch = clamp(self.pitch + dy * 0.005, 0.08, 1.50)

    def zoom(self, delta):
        self.dist = clamp(self.dist * (1.12 if delta > 0 else 0.89),
                          self.min_dist, self.max_dist)

    def set_fixed(self, pos, forward, fov=None):
        """Фиксированный режим (бортовая камера трамвайчика)."""
        self.mode = "fixed"
        self.pos = np.asarray(pos, dtype=np.float64)
        self.fwd = normalize(forward)
        if fov is not None:
            self.fov = fov

    def compute(self, surface_w, surface_h):
        """Возвращает CamData с базисом и фокусным расстоянием под размер вьюпорта."""
        if self.mode == "orbit":
            cp, sp = math.cos(self.pitch), math.sin(self.pitch)
            off = np.array([self.dist * cp * math.sin(self.yaw),
                            self.dist * sp,
                            self.dist * cp * math.cos(self.yaw)])
            self.pos = self.target + off
            fwd = normalize(self.target - self.pos)
        else:
            fwd = normalize(self.fwd)
        u = np.cross(UP, fwd)
        if float(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) < 1e-8:
            u = np.array([1.0, 0.0, 0.0])
        u = normalize(u)
        v = np.cross(fwd, u)
        focal = 0.5 * surface_h / math.tan(math.radians(self.fov) * 0.5)
        return CamData(self.pos.copy(), u, v, fwd, focal)
