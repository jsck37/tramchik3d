# -*- coding: utf-8 -*-
"""Пешеходы: блуждающие агенты с анимацией шага и 15 ключевыми точками (позы).

Состояние «сбит» (knocked): пешеход лежит PED_DOWN_TIME секунд (скелет лежачий),
затем встаёт и убегает прочь от трамвайчика."""
import math
import numpy as np
from engine.vecmath import clamp, lerp_angle, wrap_angle

import config

# Индексы ключевых точек: 0 голова, 1 шея, 2/3 плечи, 4/5 локти, 6/7 кисти,
# 8 таз, 9/10 бёдра, 11/12 колени, 13/14 стопы
LINKS = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
         (1, 8), (8, 9), (8, 10), (9, 11), (10, 12), (11, 13), (12, 14)]

_nid = [0]


def _avoid_vehicle(ped, veh, desired, w, reach):
    """Обход состава/робота II вбок: чем ближе, тем сильнее отталкивание."""
    best_d = 1e9
    bx = bz = 0.0
    sdx = sdz = 0.0
    for (x0, z0, x1, z1, hw) in veh.segments():
        dx, dz = x1 - x0, z1 - z0
        L2 = dx * dx + dz * dz
        t = 0.0 if L2 < 1e-9 else clamp(
            ((ped.pos[0] - x0) * dx + (ped.pos[1] - z0) * dz) / L2, 0.0, 1.0)
        px_, pz_ = x0 + t * dx, z0 + t * dz
        dd = math.hypot(ped.pos[0] - px_, ped.pos[1] - pz_)
        if dd < best_d:
            best_d = dd
            bx, bz = px_, pz_
            L = math.sqrt(L2)
            if L > 1e-6:
                sdx, sdz = dx / L, dz / L
    if best_d < reach:
        ax_ = ped.pos[0] - bx
        az_ = ped.pos[1] - bz
        along = ax_ * sdx + az_ * sdz            # компонента вдоль состава
        latx = ax_ - along * sdx                 # вбок от пути
        latz = az_ - along * sdz
        ll = math.hypot(latx, latz)
        if ll < 0.35:                            # ровно по курсу — в сторону
            side = 1.0 if ped.id % 2 == 0 else -1.0
            latx, latz = -sdz * side, sdx * side
            ll = 1.0
        desired = desired + np.array([latx / ll, latz / ll]) * (reach - best_d) * w
    return desired


class Pedestrian:
    def __init__(self, world, x, z, color):
        self.id = _nid[0]
        _nid[0] += 1
        self.world = world
        self.pos = np.array([float(x), float(z)])
        self.yaw = 0.0
        self.speed = 0.0
        self.max_speed = 0.9 + 0.5 * ((self.id * 37) % 7) / 7.0
        self.target = None
        self.phase = (self.id * 1.7) % 6.283
        self.color = color
        self.head_color = (232, 196, 160)
        self.knocked = 0.0        # >0 — сбит: лежит, потом встаёт и убегает
        self.flee_from = None
        self.roam = (-12.0, -5.2, 12.5, 5.2)     # прямоугольник хождения (v1.6.0)
        self.break_spot = np.array([10.4, 4.7])  # куда уходит на перерыв
        self._chk_pos = self.pos.copy()
        self._chk_t = 0.0
        self._pick_target()

    # ------------------------------------------------------------------ ИИ
    def _pick_target(self):
        x0, z0, x1, z1 = self.roam
        for _ in range(40):
            x = self.world.rng.uniform(x0, x1)
            z = self.world.rng.uniform(z0, z1)
            if self.world.point_free(x, z, 1.2) and \
               math.hypot(x - self.pos[0], z - self.pos[1]) > 4.0:
                self.target = np.array([x, z])
                return
        self.target = np.array([(x0 + x1) / 2, (z0 + z1) / 2])

    def knock_down(self, tram_pos):
        """Наезд: пешеход падает; после подъёма убегает от места удара."""
        if self.knocked > 0:
            return
        self.knocked = config.PED_DOWN_TIME
        self.speed = 0.0
        self.flee_from = np.array(tram_pos, dtype=np.float64)

    def _flee_target(self):
        """Точка в стороне от места наезда (внутри своего роам-прямоугольника)."""
        f = self.flee_from
        x0, z0, x1, z1 = self.roam
        for _ in range(40):
            ang = self.world.rng.uniform(0, 2 * math.pi)
            d = self.world.rng.uniform(6.0, 10.0)
            x = self.pos[0] + math.sin(ang) * d + (self.pos[0] - f[0]) * 0.6
            z = self.pos[1] + math.cos(ang) * d + (self.pos[1] - f[1]) * 0.6
            x = max(x0, min(x1, x))
            z = max(z0, min(z1, z))
            if self.world.point_free(x, z, 1.2):
                self.target = np.array([x, z])
                return
        self._pick_target()

    def update(self, dt):
        if self.knocked > 0:
            self.knocked -= dt
            self.speed = 0.0
            if self.knocked <= 0.0:
                self.knocked = 0.0
                self._flee_target()
            return
        # ОБЕДЕННЫЙ ПЕРЕРЫВ (наблюдение): работники отходят к рем. столу и стоят
        if self.world.break_active:
            to = self.break_spot - self.pos
            d = math.hypot(to[0], to[1])
            if d > 0.9:
                self.yaw = math.atan2(to[0], to[1])
                self.speed = self.max_speed * 0.7
                self.pos += to / d * self.speed * dt
                self.phase += self.speed * dt * 3.2
            else:
                self.speed = 0.0
            return
        to = self.target - self.pos
        d = math.hypot(to[0], to[1])
        if d < 0.9:
            self._pick_target()
            return
        # расталкивание с другими пешеходами
        sep = np.array([0.0, 0.0])
        for other in self.world.pedestrians:
            if other is self:
                continue
            diff = self.pos - other.pos
            dd = math.hypot(diff[0], diff[1])
            if 1e-4 < dd < 1.5:
                sep += diff / dd * (1.5 - dd)
        desired = to / d + sep * 0.8
        # остерегаться состава и робота II: движущийся — обходят сторо́кой,
        # стоящий — спокойно (обход вбок от габарита)
        tram = getattr(self.world, "tram", None)
        if tram is not None:
            w, reach = (1.6, 2.6) if tram.speed > 0.8 else (0.35, 2.2)
            desired = _avoid_vehicle(self, tram, desired, w, reach)
        # робот II — тоже обходят (v1.5.0)
        npc = getattr(self.world, "npc", None)
        if npc is not None:
            w, reach = (1.2, 2.4) if npc.speed > 0.3 else (0.3, 1.8)
            desired = _avoid_vehicle(self, npc, desired, w, reach)
        nd = math.hypot(desired[0], desired[1])
        if nd > 1e-4:
            desired /= nd
            self.yaw = lerp_angle(self.yaw, math.atan2(desired[0], desired[1]), 6.0 * dt)
        self.speed = self.max_speed * min(1.0, d / 2.0)
        self.pos += desired * self.speed * dt
        # границы своего роам-прямоугольника (склад A / склад B / территория)
        rx0, rz0, rx1, rz1 = self.roam
        self.pos[0] = max(rx0, min(rx1, self.pos[0]))
        self.pos[1] = max(rz0, min(rz1, self.pos[1]))
        # выталкивание из барьеров/ограждений (жёсткие прямоугольники мира)
        for (bx0, bz0, bx1, bz1, _kind, _key) in self.world.solid_rects:
            if bx1 - bx0 > 60.0 or bz1 - bz0 > 60.0:
                continue                       # стены-периметр — это roam
            m = 0.45
            if bx0 - m < self.pos[0] < bx1 + m and bz0 - m < self.pos[1] < bz1 + m:
                dl = self.pos[0] - (bx0 - m)
                dr = (bx1 + m) - self.pos[0]
                db = self.pos[1] - (bz0 - m)
                df = (bz1 + m) - self.pos[1]
                mmp = min(dl, dr, db, df)
                if mmp == dl:
                    self.pos[0] -= dl
                elif mmp == dr:
                    self.pos[0] += dr
                elif mmp == db:
                    self.pos[1] -= db
                else:
                    self.pos[1] += df
        # выталкивание из ящиков
        for cr in self.world.crates:
            diff = self.pos - cr["pos"]
            dd = math.hypot(diff[0], diff[1])
            rr = cr["radius"] + 0.45
            if 1e-4 < dd < rr:
                self.pos += diff / dd * (rr - dd)
        # выталкивание из рем. столов
        for tb in getattr(self.world, "tables", []):
            dx0 = self.pos[0] - tb["pos"][0]
            dz0 = self.pos[1] - tb["pos"][1]
            if abs(dx0) < tb["size"][0] / 2 + 0.5 and \
               abs(dz0) < tb["size"][1] / 2 + 0.5:
                px_ = math.copysign(tb["size"][0] / 2 + 0.5 - abs(dx0), dx0)
                pz_ = math.copysign(tb["size"][1] / 2 + 0.5 - abs(dz0), dz0)
                if abs(px_) < abs(pz_):
                    self.pos[0] += px_
                else:
                    self.pos[1] += pz_
        # выталкивание из габарита состава (нельзя проходить сквозь вагоны)
        tram = getattr(self.world, "tram", None)
        if tram is not None:
            for (x0, z0, x1, z1, hw) in tram.segments():
                dx, dz = x1 - x0, z1 - z0
                L2 = dx * dx + dz * dz
                t = 0.0 if L2 < 1e-9 else clamp(
                    ((self.pos[0] - x0) * dx + (self.pos[1] - z0) * dz) / L2, 0.0, 1.0)
                px_, pz_ = x0 + t * dx, z0 + t * dz
                diff = self.pos - np.array([px_, pz_])
                dd = math.hypot(diff[0], diff[1])
                rr = hw + 0.55
                if 1e-4 < dd < rr:
                    self.pos += diff / dd * (rr - dd)
        # антизастревание: нет прогресса 1.5 с — берём новую цель
        self._chk_t += dt
        if self._chk_t >= 1.5:
            if math.hypot(*(self.pos - self._chk_pos)) < 0.3:
                self._pick_target()
            self._chk_pos = self.pos.copy()
            self._chk_t = 0.0
        self.phase += self.speed * dt * 3.2

    # --------------------------------------------------- геометрия и позы
    def fwd(self):
        return np.array([math.sin(self.yaw), math.cos(self.yaw)])

    def right(self):
        return np.array([math.cos(self.yaw), -math.sin(self.yaw)])

    def _p3(self, local_right, up, local_fwd):
        """Локальные координаты → мировые (x, y, z)."""
        r = self.right()
        f = self.fwd()
        return np.array([self.pos[0] + r[0] * local_right + f[0] * local_fwd,
                         up,
                         self.pos[1] + r[1] * local_right + f[1] * local_fwd])

    def keypoints3d(self):
        """15 ключевых точек скелета (масштаб ~ COCO)."""
        if self.knocked > 0:
            return self._lying_keypoints()
        swing = math.sin(self.phase) * 0.26
        sway = math.cos(self.phase) * 0.03
        return [
            self._p3(sway, 1.72, 0.02),          # 0 голова (верх)
            self._p3(sway, 1.50, 0.0),           # 1 шея
            self._p3(0.26 + sway, 1.44, 0.0),    # 2 плечо L
            self._p3(-0.26 + sway, 1.44, 0.0),   # 3 плечо R
            self._p3(0.33 - swing * 0.5, 1.06, swing * 0.5),   # 4 локоть L
            self._p3(-0.33 + swing * 0.5, 1.06, -swing * 0.5),  # 5 локоть R
            self._p3(0.36 - swing * 0.7, 0.74, swing * 0.9),   # 6 кисть L
            self._p3(-0.36 + swing * 0.7, 0.74, -swing * 0.9),  # 7 кисть R
            self._p3(sway, 0.94, 0.0),           # 8 таз
            self._p3(0.14, 0.90, 0.0),           # 9 бедро L
            self._p3(-0.14, 0.90, 0.0),          # 10 бедро R
            self._p3(0.15, 0.48, swing),         # 11 колено L
            self._p3(-0.15, 0.48, -swing),       # 12 колено R
            self._p3(0.15, 0.05, swing * 1.8),   # 13 стопа L
            self._p3(-0.15, 0.05, -swing * 1.8),  # 14 стопа R
        ]

    def _lying_keypoints(self):
        """Скелет лежащего на спине пешехода (голова по курсу падения)."""
        return [
            self._p3(0.0, 0.30, 0.55),     # 0 голова
            self._p3(0.0, 0.26, 0.30),     # 1 шея
            self._p3(0.26, 0.26, 0.30),    # 2 плечо L
            self._p3(-0.26, 0.26, 0.30),   # 3 плечо R
            self._p3(0.32, 0.14, 0.08),    # 4 локоть L
            self._p3(-0.32, 0.14, 0.08),   # 5 локоть R
            self._p3(0.34, 0.10, -0.16),   # 6 кисть L
            self._p3(-0.34, 0.10, -0.16),  # 7 кисть R
            self._p3(0.0, 0.22, 0.05),     # 8 таз
            self._p3(0.15, 0.20, -0.12),   # 9 бедро L
            self._p3(-0.15, 0.20, -0.12),  # 10 бедро R
            self._p3(0.17, 0.16, -0.55),   # 11 колено L
            self._p3(-0.17, 0.16, -0.55),  # 12 колено R
            self._p3(0.19, 0.10, -0.95),   # 13 стопа L
            self._p3(-0.19, 0.10, -0.95),  # 14 стопа R
        ]
