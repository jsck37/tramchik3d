# -*- coding: utf-8 -*-
"""Штабелёры склада (наблюдение: «груз ставят на обычный пол зон
погрузки/выгрузки, а оттуда работники убирают его на штабелерах»).

- ШТАБЕЛЁР A (склад погрузки): возит бигбоксы от стеллажного запаса
  к площадке ПОГРУЗКИ, пока состав стоит на погрузке;
- ШТАБЕЛЁР B (склад выгрузки): убирает бигбоксы с площадки РАЗГРУЗКИ
  к стеллажам после разгрузки состава;
- анимация: подъезд → подъём вил → перевозка → опускание; simple FSM.
"""
import math

import numpy as np

import config
from engine.mesh import box_mesh, cylinder_mesh, merge_parts

PAD = config.PAL

# ключевые точки (склад A; для B — зеркалирование по X: x' = 109 - x)
PICK_A = (-4.1, 1.2)          # у стеллажного запаса (гнёзда с бигбоксами)
HOME_A = (-5.9, -3.6)         # стоянка
PAD_A = (10.6, -1.9)          # площадка погрузки (гнёзда)
RACK_A = (-8.2, -3.3)         # точка разгрузки у острова
PICK_B = (109.0 - PICK_A[0], 0.0)
HOME_B = (109.0 - HOME_A[0], -3.6)
PAD_B = (98.4, 1.9)           # площадка разгрузки (гнёзда)
RACK_B = (109.0 - RACK_A[0], -3.3)

LANE_B_MID = (108.6, -3.3)    # обход острова B по северной стороне


def _mesh():
    parts = [
        box_mesh(0.86, 0.5, 1.15, (196, 118, 44), 0, 0.35, 0.12),
        box_mesh(0.9, 0.1, 1.2, (40, 44, 52), 0, 0.12, 0.12),
        box_mesh(0.8, 0.06, 0.8, (60, 66, 80), 0, 1.62, 0.22),
        # стойки защитной крыши
        box_mesh(0.07, 0.78, 0.07, (60, 66, 80), 0.36, 1.22, 0.5),
        box_mesh(0.07, 0.78, 0.07, (60, 66, 80), -0.36, 1.22, 0.5),
        # мачта (2 стойки) + каретка
        box_mesh(0.09, 2.0, 0.09, (70, 76, 92), 0.30, 1.0, -0.48),
        box_mesh(0.09, 2.0, 0.09, (70, 76, 92), -0.30, 1.0, -0.48),
        box_mesh(0.66, 0.12, 0.10, (110, 118, 136), 0, 0.5, -0.48),
        # вилы (2 зуба вперёд)
        box_mesh(0.12, 0.05, 0.95, (150, 156, 172), 0.22, 0.12, -0.95),
        box_mesh(0.12, 0.05, 0.95, (150, 156, 172), -0.22, 0.12, -0.95),
        # водитель (упрощённый)
        box_mesh(0.4, 0.5, 0.28, (60, 130, 190), 0, 0.95, 0.3),
        box_mesh(0.22, 0.24, 0.22, (232, 196, 160), 0, 1.32, 0.3),
        # маячок
        cylinder_mesh(0.06, 0.1, 6, (255, 210, 50), axis="y", cy=1.7, cz=0.22),
        # противовес
        box_mesh(0.8, 0.34, 0.24, (40, 44, 52), 0, 0.42, 0.66),
    ]
    return merge_parts([(mm, (0, 0, 0), 0.0) for mm in parts])


def _crate_mesh():
    return merge_parts([
        (box_mesh(0.78, 0.62, 0.7, PAD["bigbox"], 0, 0.31, 0), (0, 0, 0), 0.0),
    ])


class Stacker:
    """Штабелёр одного склада: side = "A" (погрузка) | "B" (выгрузка)."""

    def __init__(self, world, side):
        self.world = world
        self.side = side
        self.name = "Штабелёр A" if side == "A" else "Штабелёр B"
        self.pos = np.array(HOME_A if side == "A" else HOME_B, dtype=np.float64)
        self.yaw = 0.0
        self.state = "IDLE"
        self.timer = 0.0
        self.path = []
        self.pi = 0
        self.carrying = False
        self.fork_lift = 0.0          # 0..1 — высота вил/груза
        self.m_mesh = _mesh()
        self.c_mesh = _crate_mesh()
        self.speed = 1.15             # м/с (медленнее состава)
        self._done = 0                # сколько убрано (для статистики)

    # ---------------------------------------------------------------- утилиты
    def _wp(self, name):
        if self.side == "A":
            return {"pick": PICK_A, "home": HOME_A, "pad": PAD_A,
                    "rack": RACK_A}[name]
        return {"pick": PICK_B, "home": HOME_B, "pad": PAD_B,
                "rack": RACK_B}[name]

    def _go(self, pts):
        self.path = [(float(p[0]), float(p[1])) for p in pts]
        self.pi = 0

    # ---------------------------------------------------------------- логика
    def update(self, dt):
        tr = getattr(self.world, "tram", None)
        loading = tr is not None and tr.state == "ПОГРУЗКА"
        unloaded = any(c.get("staged") for c in self.world.crates)
        lifted = 0.35 if self.carrying else 0.0
        self.fork_lift += (lifted - self.fork_lift) * min(1.0, 5.0 * dt)

        if self.state == "IDLE":
            if self.side == "A" and loading and not self.carrying:
                # ехать за бигбоксом к стеллажному запасу
                self._go([self._wp("pick")])
                self.state = "TO_PICK"
            elif self.side == "B" and unloaded and not self.carrying:
                # забрать первый бигбокс с площадки
                staged = [c for c in self.world.crates if c.get("staged")]
                if staged:
                    c0 = staged[0]
                    self._go([(c0["pos"][0], c0["pos"][1])])
                    self.state = "TO_PICK"
        elif self.state == "TO_PICK":
            if self._follow(dt):
                self.state = "PICK"
                self.timer = 0.9
        elif self.state == "PICK":
            self.timer -= dt
            if self.timer <= 0.0:
                if self.side == "B":
                    staged = [c for c in self.world.crates if c.get("staged")]
                    if staged:
                        self.world.crates.remove(staged[0])
                self.carrying = True
                target = (self._wp("rack") if self.side == "B"
                          else self._wp("pad"))
                mid = LANE_B_MID if self.side == "B" else None
                pts = [mid] if mid else []
                self._go(pts + [target])
                self.state = "TO_DROP"
        elif self.state == "TO_DROP":
            if self._follow(dt):
                self.state = "DROP"
                self.timer = 0.9
        elif self.state == "DROP":
            self.timer -= dt
            if self.timer <= 0.0:
                self.carrying = False
                self._done += 1
                if self.side == "A" and tr is not None:
                    tr.stacker_delivered = getattr(tr, "stacker_delivered", 0) + 1
                self._go([self._wp("home")])
                self.state = "RETURN"
        elif self.state == "RETURN":
            if self._follow(dt):
                self.state = "IDLE"

    def _follow(self, dt):
        """Движение по точкам; True — точка/маршрут завершены."""
        if self.pi >= len(self.path):
            return True
        tx, tz = self.path[self.pi]
        dx, dz = tx - self.pos[0], tz - self.pos[1]
        d = math.hypot(dx, dz)
        if d < 0.12:
            self.pi += 1
            return self.pi >= len(self.path)
        desired = math.atan2(dx, dz)
        diff = (desired - self.yaw + math.pi) % math.tau - math.pi
        self.yaw += max(-2.6 * dt, min(2.6 * dt, diff * 4.0 * dt))
        step = min(self.speed * dt, d)
        self.pos[0] += math.sin(self.yaw) * step
        self.pos[1] += math.cos(self.yaw) * step
        return False

    # ---------------------------------------------------------------- рендер
    def draw(self, r):
        r.draw_mesh(self.m_mesh, (self.pos[0], 0.0, self.pos[1]), self.yaw)
        if self.carrying:
            lift = self.fork_lift * 0.55
            r.draw_mesh(self.c_mesh, (self.pos[0] - math.sin(self.yaw) * 0.95,
                                      lift,
                                      self.pos[1] - math.cos(self.yaw) * 0.95),
                        self.yaw)
