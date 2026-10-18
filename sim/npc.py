# -*- coding: utf-8 -*-
"""Робот II — второй состав склада (наблюдение: «2-й робот прибыл,
ручное управление», «ожидание выезда 1-го робота»).

- ездит по малому кольцу вокруг стеллажных ОСТРОВОВ склада A
  (западная половина, вне зоны погрузки и малого круга игрока);
- управляется оператором вручную → на обеденном перерыве стоит;
- уступает дорогу составу игрока: если состав впереди по курсу — стоит
  и ждёт (как в наблюдении); после долгого ожидания оператор медленно
  объезжает (анти-дедлок), вплотную к составу — жёсткая остановка;
- столкновение с ним — штраф −30 (учитывается у игрока).
"""
import math
from collections import deque

import numpy as np

import config
from sim.tram import build_loco_mesh, build_wagon_mesh
from engine.vecmath import wrap_angle, clamp

PAD = config.PAL

# кольцевой маршрут робота II — вокруг островов склада A (западная половина)
ROUTE = [(-3.9, 3.55), (-11.6, 3.55), (-11.6, -3.55), (-3.9, -3.55)]
PARK_PAUSE = 10.0       # стоянка в стартовой точке, сек
YIELD_DIST = config.NPC_YIELD_DIST
CRAWL_SPEED = config.NPC_CRAWL
WAIT_NUDGE = config.NPC_WAIT_NUDGE
HARD_STOP = config.NPC_HARD_STOP


class NPCTram:
    def __init__(self, world):
        self.world = world
        self.name = "Робот II"
        self.pos = np.array([ROUTE[0][0], ROUTE[0][1]])
        self.yaw = math.atan2(ROUTE[1][0] - ROUTE[0][0],
                              ROUTE[1][1] - ROUTE[0][1])   # курс на след. точку
        self.speed = 0.0
        self.wi = 1                       # индекс следующей точки маршрута
        self.pause = PARK_PAUSE
        self.brake_t = 0.0                # форсированная остановка (после удара)
        self.wait_t = 0.0                 # сколько ждёт состав игрока (анти-дедлок)
        self.n_wagons = config.N_WAGONS
        self.wagon_pos = [self.pos.copy() for _ in range(self.n_wagons)]
        self.wagon_yaw = [self.yaw] * self.n_wagons
        self.trail = deque(maxlen=2000)
        self._seed_trail()
        # меши: серо-синий корпус, синий маячок (отличие от игрока)
        self.m_loco = build_loco_mesh((124, 134, 168), (150, 160, 196),
                                      (90, 160, 255))
        self.m_wag = [build_wagon_mesh((110, 120, 156)) for _ in range(3)]

    # ------------------------------------------------------------------ геом.
    def fwd(self):
        return np.array([math.sin(self.yaw), math.cos(self.yaw)])

    def box_corners(self, center2d, cy0, cy1, half_w, half_l, yaw):
        """8 углов бокса (для восприятия — как у состава игрока)."""
        s, c = math.sin(yaw), math.cos(yaw)
        out = []
        for (lx, ly, lz) in ((-half_w, cy0, -half_l), (half_w, cy0, -half_l),
                             (half_w, cy0, half_l), (-half_w, cy0, half_l),
                             (-half_w, cy1, -half_l), (half_w, cy1, -half_l),
                             (half_w, cy1, half_l), (-half_w, cy1, half_l)):
            out.append(np.array([center2d[0] + lx * c + lz * s, ly,
                                 center2d[1] - lx * s + lz * c]))
        return out

    def gaps(self):
        return [config.GAP_LOCO_WAGON] + [config.WAGON_GAP] * (self.n_wagons - 1)

    def segments(self):
        f = self.fwd()
        out = [(self.pos[0] - f[0] * 0.80, self.pos[1] - f[1] * 0.80,
                self.pos[0] + f[0] * 0.82, self.pos[1] + f[1] * 0.82,
                config.LOCO_WIDTH / 2)]
        for wp, wy in zip(self.wagon_pos, self.wagon_yaw):
            fw = (math.sin(wy), math.cos(wy))
            out.append((wp[0] - fw[0] * 0.425, wp[1] - fw[1] * 0.425,
                        wp[0] + fw[0] * 0.425, wp[1] + fw[1] * 0.425,
                        config.WAGON_WIDTH / 2))
        return out

    def brake(self, t):
        """Форсированная остановка (после столкновения)."""
        self.brake_t = max(self.brake_t, t)

    def _seed_trail(self):
        self.trail.clear()
        f = self.fwd()
        d = 0.0
        total = config.GAP_LOCO_WAGON + config.WAGON_GAP * (self.n_wagons - 1) + 1.5
        while d <= total:
            px = self.pos[0] - f[0] * d
            pz = self.pos[1] - f[1] * d
            self.trail.append((px, pz))
            d += 0.08
        self.trail.reverse()
        self._place_wagons()

    # ------------------------------------------------------------------ логика
    def _min_dist_to_player(self):
        """Минимальная дистанция единиц робота до кузова игрока (минус полуширина)."""
        tram = getattr(self.world, "tram", None)
        if tram is None:
            return 1e9
        best = 1e9
        pts = [self.pos] + list(self.wagon_pos)
        for (x0, z0, x1, z1, hw) in tram.segments():
            for p in pts:
                dx, dz = x1 - x0, z1 - z0
                L2 = dx * dx + dz * dz
                t = 0.0 if L2 < 1e-9 else clamp(
                    ((p[0] - x0) * dx + (p[1] - z0) * dz) / L2, 0.0, 1.0)
                best = min(best, math.hypot(p[0] - (x0 + t * dx),
                                            p[1] - (z0 + t * dz)) - hw)
        return best

    def _yield_to_player(self):
        """Уступает дорогу: состав игрока ВПЕРЕДИ по курсу → стоп.
        Проезд мимо (сбоку/сзади) не блокирует — иначе дедлоки."""
        tram = getattr(self.world, "tram", None)
        if tram is None:
            return False
        f = self.fwd()
        for (x0, z0, x1, z1, hw) in tram.segments():
            dx, dz = x1 - x0, z1 - z0
            L2 = dx * dx + dz * dz
            t = 0.0 if L2 < 1e-9 else clamp(
                ((self.pos[0] - x0) * dx + (self.pos[1] - z0) * dz) / L2, 0.0, 1.0)
            px_, pz_ = x0 + t * dx, z0 + t * dz
            diff = np.array([px_ - self.pos[0], pz_ - self.pos[1]])
            d = math.hypot(diff[0], diff[1])
            if d > YIELD_DIST:
                continue
            along = diff[0] * f[0] + diff[1] * f[1]
            if along > 1.0 and d < YIELD_DIST:   # только строго впереди
                return True
        return False

    def update(self, dt):
        self.brake_t = max(0.0, self.brake_t - dt)
        dmin = self._min_dist_to_player()
        hard = dmin < HARD_STOP               # вплотную — жёсткая остановка
        yielding = self._yield_to_player()
        if yielding and not hard:
            self.wait_t += dt
        else:
            self.wait_t = 0.0
        nudge = yielding and self.wait_t > WAIT_NUDGE and not hard
        paused = (self.pause > 0.0 or self.world.break_active or
                  self.brake_t > 0.0 or hard)
        if paused:
            self.speed = max(0.0, self.speed - 1.8 * dt)
            if self.pause > 0.0:
                self.pause -= dt
        elif yielding and not nudge:
            self.speed = max(0.0, self.speed - 2.2 * dt)   # тормозит и ждёт
        else:
            tx, tz = ROUTE[self.wi]
            desired = math.atan2(tx - self.pos[0], tz - self.pos[1])
            diff = wrap_angle(desired - self.yaw)
            self.yaw += clamp(diff * 2.0, -1.2, 1.2) * dt
            base = CRAWL_SPEED if nudge else config.NPC_SPEED
            target = base * (0.45 if abs(diff) > 0.5 else 1.0)
            if target > self.speed:
                self.speed = min(target, self.speed + 1.2 * dt)
            else:
                self.speed = max(target, self.speed - 1.8 * dt)
            d = math.hypot(tx - self.pos[0], tz - self.pos[1])
            if d < 0.7:
                self.wi = (self.wi + 1) % len(ROUTE)
                if self.wi == 0:
                    self.pause = PARK_PAUSE      # стоянка в начале круга
        if self.speed > 0.0:
            f = self.fwd()
            self.pos = self.pos + f * self.speed * dt
        # след
        last = self.trail[-1] if self.trail else None
        if last is None or math.hypot(self.pos[0] - last[0],
                                      self.pos[1] - last[1]) >= 0.08:
            self.trail.append((float(self.pos[0]), float(self.pos[1])))
        self._place_wagons()

    def _place_wagons(self):
        n = self.n_wagons
        if not self.trail:
            return
        gaps = self.gaps()
        need = gaps[0]
        acc = 0.0
        wi = 0
        prev = None
        for p in reversed(self.trail):
            if prev is not None:
                d = math.hypot(prev[0] - p[0], prev[1] - p[1])
                while wi < n and acc + d >= need - 1e-9:
                    t = (need - acc) / max(d, 1e-9)
                    wx = p[0] + (prev[0] - p[0]) * t
                    wz = p[1] + (prev[1] - p[1]) * t
                    self.wagon_pos[wi] = np.array([wx, wz])
                    self.wagon_yaw[wi] = math.atan2(prev[0] - p[0], prev[1] - p[1])
                    wi += 1
                    need += gaps[min(wi, len(gaps) - 1)]
                acc += d
                if wi >= n:
                    break
            prev = p
        while wi < n:
            last = self.wagon_pos[wi - 1] if wi > 0 else self.pos
            lyaw = self.wagon_yaw[wi - 1] if wi > 0 else self.yaw
            self.wagon_pos[wi] = np.array([last[0], last[1]])
            self.wagon_yaw[wi] = lyaw
            wi += 1

    # ------------------------------------------------------------------ рендер
    def draw(self, r):
        r.draw_mesh(self.m_loco, (self.pos[0], 0.0, self.pos[1]), self.yaw)
        for i in range(self.n_wagons):
            wp = self.wagon_pos[i]
            r.draw_mesh(self.m_wag[i % len(self.m_wag)],
                        (wp[0], 0.0, wp[1]), self.wagon_yaw[i])
        pts = [(self.pos[0], self.pos[1])] + \
              [(wp[0], wp[1]) for wp in self.wagon_pos]
        segs = [(pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1])
                for k in range(len(pts) - 1)]
        r.draw_lines(segs, (30, 32, 38), width=2, y=0.30)
