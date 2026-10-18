# -*- coding: utf-8 -*-
"""Трамвайчик — робот-тягач RoboCV.
"""
import math
from collections import deque

import numpy as np

import config
from engine.vecmath import wrap_angle, clamp
from engine.mesh import box_mesh, cylinder_mesh, merge_parts

PAD = config.PAL

IDLE = "IDLE"
STOP_LINE = "СТОП-ЛИНИЯ"
LOADING = "ПОГРУЗКА"
TO_B = "ДВИЖЕНИЕ К B"
UNLOADING = "РАЗГРУЗКА"
TO_A = "ВОЗВРАЩЕНИЕ К A"


def point_seg_dist(px, pz, x0, z0, x1, z1):
    """Расстояние от точки (px, pz) до отрезка (x0..x1, z0..z1) на плоскости."""
    dx, dz = x1 - x0, z1 - z0
    L2 = dx * dx + dz * dz
    if L2 < 1e-9:
        return math.hypot(px - x0, pz - z0)
    t = clamp(((px - x0) * dx + (pz - z0) * dz) / L2, 0.0, 1.0)
    return math.hypot(px - (x0 + t * dx), pz - (z0 + t * dz))


# ================================================================== меши
def build_loco_mesh(body_color, cab_color, beacon_col):
    """Тягач (паровозик) в РЕАЛЬНОМ габарите 1,6 × 0,8 × 2,2 м.
    Локальные координаты: +Z вперёд, +X вправо. Один слитый меш."""
    W, L, H = config.LOCO_WIDTH, config.LOCO_LENGTH, config.LOCO_HEIGHT
    parts = [
        # рама
        box_mesh(W - 0.02, 0.22, L - 0.05, PAD["tram_dark"], 0, 0.25, 0),
        # котёл
        cylinder_mesh(0.26, L * 0.60, 10, body_color, axis="z", cy=0.74, cz=0.20),
        cylinder_mesh(0.27, 0.16, 10, PAD["tram_dark"], axis="z", cy=0.74, cz=0.70),
        # будка
        box_mesh(W, 0.92, 0.62, cab_color, 0, 1.02, -0.40),
        box_mesh(W + 0.06, 0.07, 0.70, PAD["tram_dark"], 0, 1.50, -0.40),
        # окно будки
        box_mesh(0.60, 0.34, 0.05, (70, 140, 200), 0, 1.22, -0.08),
        # труба + колпак
        cylinder_mesh(0.09, 0.40, 8, PAD["tram_dark"], axis="y", cy=1.16, cz=0.42),
        cylinder_mesh(0.13, 0.07, 8, (90, 96, 110), axis="y", cy=1.40, cz=0.42),
        # паровой колпак
        cylinder_mesh(0.12, 0.13, 8, (225, 228, 235), axis="y", cy=1.04, cz=0.08),
        # бампер и сцепка
        box_mesh(W, 0.12, 0.08, PAD["tram_dark"], 0, 0.30, L / 2 - 0.02),
        box_mesh(0.10, 0.10, 0.22, (60, 64, 74), 0, 0.34, -L / 2 - 0.06),
    ]
    # колёса
    for oz in (0.48, -0.44):
        for ox in (W / 2 - 0.07, -(W / 2 - 0.07)):
            parts.append(cylinder_mesh(0.14, 0.07, 8, PAD["wheel"], axis="x",
                                       cx=ox, cy=0.14, cz=oz))
    # мачта ВЕРХНЕГО лидара (до 2,2 м) + купол
    parts.append(cylinder_mesh(0.035, 0.58, 6, PAD["tram_dark"], axis="y",
                               cy=1.79, cz=-0.35))
    parts.append(cylinder_mesh(0.10, 0.12, 8, (90, 210, 255), axis="y",
                               cy=2.14, cz=-0.35))
    # датчик НИЖНЕГО лидара в бампере
    parts.append(box_mesh(0.20, 0.09, 0.10, (170, 255, 120), 0, 0.16, L / 2 - 0.04))
    # маячок
    parts.append(box_mesh(0.16, 0.14, 0.16, beacon_col, 0, 1.60, -0.40))
    return merge_parts([(mm, (0, 0, 0), 0.0) for mm in parts])


def build_wagon_mesh(body_color):
    """Вагон-тележка 0,85 × 0,7 м (синяя тележка) — один слитый меш."""
    W, L = config.WAGON_WIDTH, config.WAGON_LENGTH
    parts = [
        box_mesh(W, 0.08, L, PAD["tram_dark"], 0, 0.18, 0),              # рама
        box_mesh(W - 0.04, 0.20, 0.06, body_color, 0, 0.32, L / 2 - 0.03),
        box_mesh(W - 0.04, 0.20, 0.06, body_color, 0, 0.32, -L / 2 + 0.03),
        box_mesh(0.06, 0.20, L - 0.12, body_color, W / 2 - 0.03, 0.32, 0),
        box_mesh(0.06, 0.20, L - 0.12, body_color, -(W / 2 - 0.03), 0.32, 0),
        box_mesh(0.05, 0.05, 0.24, (60, 64, 74), 0, 0.30, L / 2 + 0.10),  # дышло
    ]
    for oz in (L / 2 - 0.14, -L / 2 + 0.14):
        for ox in (W / 2 - 0.06, -(W / 2 - 0.06)):
            parts.append(cylinder_mesh(0.09, 0.05, 8, PAD["wheel"], axis="x",
                                       cx=ox, cy=0.09, cz=oz))
    return merge_parts([(mm, (0, 0, 0), 0.0) for mm in parts])


def build_cargo_mesh(kind):
    """Груз на вагоне: бигбокс / поддон с коробками / ящик.
    Груз ШИРЕ вагона на CARGO_OVERHANG с каждой стороны."""
    ow = config.CARGO_OVERHANG
    w = config.WAGON_WIDTH + 2 * ow          # выступ за габарит
    if kind == "bigbox":
        parts = [box_mesh(w, 0.50, 0.72, PAD["bigbox"], 0, 0.55, 0),
                 box_mesh(w - 0.06, 0.06, 0.66, (40, 80, 160), 0, 0.83, 0)]
    elif kind == "pallet":
        parts = [box_mesh(w, 0.12, 0.78, PAD["pallet"], 0, 0.40, 0),
                 box_mesh(w - 0.10, 0.34, 0.55, PAD["crate"][1], 0, 0.62, 0),
                 box_mesh(w - 0.14, 0.24, 0.42, PAD["crate"][0], 0.03, 0.91, 0.04)]
    else:  # ящик
        parts = [box_mesh(w, 0.10, 0.78, PAD["pallet"], 0, 0.40, 0),
                 box_mesh(w - 0.08, 0.42, 0.60, PAD["crate"][0], 0, 0.66, 0)]
    return merge_parts([(mm, (0, 0, 0), 0.0) for mm in parts])


CARGO_KINDS = ["bigbox", "pallet", "crate"]


class Tram:
    def __init__(self, world):
        self.world = world
        self.pos = np.array([config.ZONE_A[0], config.ZONE_A[1]])
        self.yaw = config.ZONE_A_YAW         # курс на восток (зона A)
        self.speed = 0.0
        self.auto = True
        self.state = STOP_LINE
        self.timer = config.STOP_LINE_TIME
        self.progress = 0.0
        self.cargo = False
        self.cargo_count = 0                 # 0..3 грузов на вагонах
        self.cycles = 0
        self.bigboxes = 0                    # отправлено бигбоксов
        self.battery = 100.0                 # АКБ (набл.: хватает на 1,5 смены)
        self._battery_warned = False
        self.path = []           # [(x, z, cap)] — фиксированный маршрут миссии
        self.pi = 0
        self.at = "A"            # у какой площадки находится состав
        self.warn = False
        self.beacon_on = False
        self._beacon_t = 0.0
        self.reverse_hold = 0.0              # >0 → показать чип «РЕВЕРС ЗАПРЕЩЁН»
        self._cap = 1.0                      # ограничение скорости (люди рядом)
        self._signal_done = False

        # --- состав ---
        self.n_wagons = config.N_WAGONS
        self.wagon_pos = [self.pos.copy() for _ in range(self.n_wagons)]
        self.wagon_yaw = [self.yaw] * self.n_wagons
        self.trail = deque(maxlen=2000)      # след тягача: [(x, z)] старые → новые
        self._seed_trail()

        # --- меши (реальные габариты) ---
        self.m_loco = build_loco_mesh(PAD["tram_body"], PAD["tram_cab"],
                                      (255, 210, 50))
        m_wag = [build_wagon_mesh(col) for col in PAD["wagon_body"]]
        m_cargo = [build_cargo_mesh(k) for k in CARGO_KINDS]
        # вагон с грузом — один слитый меш (4 draw_mesh на состав)
        self.m_wag = m_wag
        self.m_wag_loaded = [merge_parts([(m_wag[i], (0, 0, 0), 0.0),
                                          (m_cargo[i % len(m_cargo)],
                                           (0, 0, 0), 0.0)])
                             for i in range(self.n_wagons)]

    # ---------------------------------------------------------------- утилиты
    def fwd(self):
        return np.array([math.sin(self.yaw), math.cos(self.yaw)])

    @property
    def trailer_pos(self):
        """Совместимость: позиция первого вагона."""
        return self.wagon_pos[0]

    def mass(self):
        """Текущая масса состава, кг (тягач + вагоны + груз)."""
        return config.train_mass(self.cargo_count, self.n_wagons)

    def gaps(self):
        """Интервалы между центрами единиц состава (тягач→вагон1 длиннее)."""
        return [config.GAP_LOCO_WAGON] + [config.WAGON_GAP] * (self.n_wagons - 1)

    def cam_pose(self):
        """Поза КАМЕРЫ 1 (передней): на мачте над котлом, чуть вниз."""
        f = self.fwd()
        pos = np.array([self.pos[0] + f[0] * 0.55, 2.35,
                        self.pos[1] + f[1] * 0.55])
        return pos, np.array([f[0], -0.10, f[1]])

    def rear_pose(self):
        """Поза КАМЕРЫ 2 (задней): на последнем вагоне, смотрит назад."""
        wp = self.wagon_pos[-1]
        wy = self.wagon_yaw[-1]
        f = (math.sin(wy), math.cos(wy))
        pos = np.array([wp[0] - f[0] * 0.5, 1.05, wp[1] - f[1] * 0.5])
        return pos, np.array([-f[0], -0.12, -f[1]])

    def box_corners(self, center2d, cy0, cy1, half_w, half_l, yaw):
        """8 углов бокса (для восприятия: рамки/сегментация)."""
        s, c = math.sin(yaw), math.cos(yaw)
        out = []
        for (lx, ly, lz) in ((-half_w, cy0, -half_l), (half_w, cy0, -half_l),
                             (half_w, cy0, half_l), (-half_w, cy0, half_l),
                             (-half_w, cy1, -half_l), (half_w, cy1, -half_l),
                             (half_w, cy1, half_l), (-half_w, cy1, half_l)):
            out.append(np.array([center2d[0] + lx * c + lz * s, ly,
                                 center2d[1] - lx * s + lz * c]))
        return out

    def segments(self):
        """Отрезки кузова состава для коллизий: (x0, z0, x1, z1, half_width)."""
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

    def cargo_segments(self):
        """Отрезки ВЫСТУПАЮЩЕГО груза на вагонах. Пусто без груза."""
        if not self.cargo or self.cargo_count <= 0:
            return []
        hw = config.WAGON_WIDTH / 2 + config.CARGO_OVERHANG
        out = []
        for wp, wy in zip(self.wagon_pos, self.wagon_yaw):
            fw = (math.sin(wy), math.cos(wy))
            out.append((wp[0] - fw[0] * 0.40, wp[1] - fw[1] * 0.40,
                        wp[0] + fw[0] * 0.40, wp[1] + fw[1] * 0.40, hw))
        return out

    def length_total(self):
        """Габаритная длина состава, м (для HUD/статистики)."""
        return (config.LOCO_LENGTH / 2 + config.GAP_LOCO_WAGON +
                (self.n_wagons - 1) * config.WAGON_GAP + config.WAGON_LENGTH / 2)

    # ------------------------------------------------------------------ сброс
    def reset(self):
        self.pos = np.array([config.ZONE_A[0], config.ZONE_A[1]])
        self.yaw = config.ZONE_A_YAW
        self.speed = 0.0
        self.state = STOP_LINE
        self.timer = config.STOP_LINE_TIME
        self.progress = 0.0
        self.cargo = False
        self.cargo_count = 0
        self.cycles = 0
        self.bigboxes = 0
        self.battery = 100.0
        self._battery_warned = False
        self.path = []
        self.pi = 0
        self.at = "A"
        self.warn = False
        self.reverse_hold = 0.0
        self._cap = 1.0
        self._signal_done = False
        self._seed_trail()

    def _seed_trail(self):
        """След позади стартовой позиции — чтобы вагоны стояли «составом»."""
        self.trail.clear()
        f = self.fwd()
        pts = []
        d = 0.0
        total = config.GAP_LOCO_WAGON + config.WAGON_GAP * (self.n_wagons - 1) + 2.0
        while d <= total:
            px = self.pos[0] - f[0] * d
            pz = self.pos[1] - f[1] * d
            if not self.world.contains(px, pz, 0.5):
                break
            pts.append((px, pz))
            d += 0.08
        pts.reverse()                        # старые (дальние) — в начало
        self.trail.extend(pts)
        self._place_wagons()

    # ------------------------------------------------------------------ цикл
    def update(self, dt, keys=None):
        self._beacon_t += dt
        self.beacon_on = int(self._beacon_t * 2.4) % 2 == 0
        self.reverse_hold = max(0.0, self.reverse_hold - dt * 0.8)
        if self.auto:
            self._auto(dt)
        else:
            self._manual(dt, keys)
        self._integrate(dt)
        # след и расстановка вагонов
        last = self.trail[-1] if self.trail else None
        if last is None or math.hypot(self.pos[0] - last[0],
                                      self.pos[1] - last[1]) >= 0.08:
            self.trail.append((float(self.pos[0]), float(self.pos[1])))
        self._place_wagons()
        self._check_pedestrians()
        self._check_walls()
        self._check_npc()

    # ------------------------------------------------------------- автопилот
    def _auto(self, dt):
        self.warn = False
        if self.state == STOP_LINE:
            # наблюдение: состав останавливается перед началом зоны (18–28 с)
            self.speed = 0.0
            self.progress = 1.0 - max(0.0, self.timer) / config.STOP_LINE_TIME
            self.timer -= dt
            if self.timer <= 0.0:
                self._signal_done = False
                if self.at == "A":           # площадка ПОГРУЗКИ склада A
                    self.state = LOADING
                    self.timer = config.LOAD_TIME
                else:                        # площадка РАЗГРУЗКИ склада B
                    self.state = UNLOADING
                    self.timer = config.UNLOAD_TIME
            return

        if self.state in (LOADING, UNLOADING):
            self.speed = 0.0
            self.timer -= dt
            if self.state == LOADING:
                total = config.LOAD_TIME
                self.progress = 1.0 - max(0.0, self.timer) / total
                self.cargo_count = int(round((1.0 - max(0.0, self.timer) / total)
                                             * self.n_wagons))
                if self.timer <= 0.0:
                    self.cargo = True
                    self.cargo_count = self.n_wagons
                    self.state = TO_B
                    self.path = self.world.mission_path("to_b")
                    self.pi = 0
            else:
                total = config.UNLOAD_TIME
                self.progress = 1.0 - max(0.0, self.timer) / total
                self.cargo_count = int(round(max(0.0, self.timer) / total
                                             * self.n_wagons))
                if self.timer <= 0.0:
                    self.cargo = False
                    self.cargo_count = 0
                    self.cycles += 1
                    self.bigboxes += config.BIGBOX_PER_TRIP
                    self.battery = max(5.0, self.battery -
                                       config.BATTERY_DRAIN_PER_TRIP)
                    if self.battery < config.BATTERY_WARN and not self._battery_warned:
                        self._battery_warned = True
                        self.world.events.append(
                            ("АКБ < 20% — ТРЕБУЕТСЯ ЗАРЯДКА У ВОРОТ", (255, 200, 80)))
                    self.world.add_score(
                        config.DELIVERY_REWARD,
                        f"Рейс выполнен (3 бигбокса): +{config.DELIVERY_REWARD}",
                        PAD["ok"])
                    self.world.on_unloaded()   # груз на пол — уберут штабелёром
                    self.world.spawn_carts()   # синие тележки — наблюдение
                    self.state = TO_A
                    self.path = self.world.mission_path("to_a")  # через малый круг B
                    self.pi = 0
            return

        # ---- движение по фиксированному маршруту миссии (150 м + круга) ----
        if not self.path:
            self.path = (self.world.mission_path("to_b") if self.at == "A"
                         else self.world.mission_path("to_a"))
            self.pi = 0
        # безопасность состава: человек вплотную — не трогаемся; рядом — тихо
        self._cap = 1.0
        # фьюжн с ВЕРХНИМ лидаром: фронтальный барьер — снижаем скорость
        lid = getattr(self.world, "lidar_top", None)
        if lid is not None and lid.scan:
            fc = lid.front_clearance(self, 30.0)
            if fc < 1.0 and self.speed > 0.1:
                self.speed = min(self.speed, 0.35)
            elif fc < 2.4:
                self._cap = min(self._cap, 0.55)
        blocked = False
        # работники
        for ped in self.world.pedestrians:
            if ped.knocked > 0:
                continue
            for (x0, z0, x1, z1, hw) in self.segments():
                d = point_seg_dist(ped.pos[0], ped.pos[1], x0, z0, x1, z1)
                if d < hw + 0.7:
                    blocked = True
                    break
                if d < hw + 2.2:
                    self._cap = 0.6
            if blocked:
                break
        # робот II — блокирует только если он ВПЕРЕДИ по курсу (наблюдение:
        # «ожидание выезда»); проезд рядом без остановки — без блока
        npc = getattr(self.world, "npc", None)
        if not blocked and npc is not None:
            f0 = self.fwd()
            loco_seg = self.segments()[0]
            for (nx0, nz0, nx1, nz1, nhw) in npc.segments():
                cx_, cz_ = (nx0 + nx1) * 0.5, (nz0 + nz1) * 0.5
                d = point_seg_dist(cx_, cz_, loco_seg[0], loco_seg[1],
                                   loco_seg[2], loco_seg[3])
                along = (cx_ - self.pos[0]) * f0[0] + (cz_ - self.pos[1]) * f0[1]
                if along > 0.3:
                    if d < loco_seg[4] + nhw + 0.8:
                        blocked = True
                        break
                    if d < loco_seg[4] + nhw + 2.4:
                        self._cap = min(self._cap, 0.6)
        if blocked:
            self.speed = 0.0
            self.warn = True
            return
        # экстренное торможение: пешеход впереди по курсу (в т.ч. лежащий)
        f = self.fwd()
        for ped in self.world.pedestrians:
            diff = ped.pos - self.pos
            d = math.hypot(diff[0], diff[1])
            if d < 4.0:
                along = (diff[0] * f[0] + diff[1] * f[1]) / max(d, 1e-4)
                if along > 0.72:
                    self.warn = d < 3.2 and ped.knocked <= 0.0
                    if d < 2.4:
                        self.progress = 0.0
                        self._steer_to_point(dt, ped.pos[0], ped.pos[1], stop=True)
                        return
        self._follow_path(dt)

    def _follow_path(self, dt):
        """Движение по точкам маршрута миссии (третий элемент — лимит скорости)."""
        if not self.path:
            self.speed = 0.0
            return
        tx, tz, wcap = self.path[min(self.pi, len(self.path) - 1)]
        d = math.hypot(tx - self.pos[0], tz - self.pos[1])
        if d < 1.1:
            if self.pi >= len(self.path) - 1:
                self._arrive()
                return
            self.pi += 1
            tx, tz, wcap = self.path[self.pi]
            d = math.hypot(tx - self.pos[0], tz - self.pos[1])
        self._cap = min(self._cap, wcap)
        self._steer_to_point(dt, tx, tz)

    def _steer_to_point(self, dt, tx, tz, stop=False):
        desired = math.atan2(tx - self.pos[0], tz - self.pos[1])
        diff = wrap_angle(desired - self.yaw)
        turn = clamp(diff * 2.4, -config.TRAM_TURN, config.TRAM_TURN)
        self.yaw += turn * dt
        target_speed = 0.0 if stop else config.TRAM_SPEED * self._cap * \
            (0.4 if abs(diff) > 0.6 else 1.0)
        accel = config.DRIVE_FORCE / self.mass()          # физика массы
        brake = config.BRAKE_FORCE / self.mass() * 1.6
        rate = brake if target_speed < self.speed else accel
        self.speed = max(0.0, self.speed + clamp(target_speed - self.speed,
                                                 -rate * dt, rate * dt))

    def _arrive(self):
        self.speed = 0.0
        self.path = []
        self.progress = 0.0
        # у какой площадки остановились (A — погрузка, B — разгрузка)
        da = math.hypot(self.pos[0] - config.ZONE_A[0],
                        self.pos[1] - config.ZONE_A[1])
        db = math.hypot(self.pos[0] - config.ZONE_B[0],
                        self.pos[1] - config.ZONE_B[1])
        self.at = "A" if da <= db else "B"
        # стоп-линия перед зоной (наблюдение: остановка перед началом з/р)
        self.state = STOP_LINE
        self.timer = config.STOP_LINE_TIME
        if not self._signal_done:
            self._signal_done = True
            self.world.events.append(
                ("СИГНАЛ: состав прибыл к зоне (звуковой сигнал)", (120, 200, 255)))

    # ------------------------------------------------------------ ручной режим
    def _manual(self, dt, keys):
        k = keys or {}
        fwd_k = k.get("up", False)
        back_k = k.get("down", False)
        left_k = k.get("left", False)
        right_k = k.get("right", False)
        # РЕВЕРС ЗАПРЕЩЁН: S/↓ и ПРОБЕЛ — только тормоз, заднего хода нет
        braking = back_k or k.get("brake", False)
        accel = config.DRIVE_FORCE / self.mass()
        brake = config.BRAKE_FORCE / self.mass() * (1.35 if back_k else 1.0)
        if braking:
            self.speed = max(0.0, self.speed - brake * dt)
            if self.speed < 0.05 and back_k:
                self.reverse_hold = 1.0
        else:
            target = config.TRAM_SPEED if fwd_k else 0.0
            rate = accel if target > self.speed else brake
            self.speed = max(0.0, self.speed + clamp(target - self.speed,
                                                     -rate * dt, rate * dt))
        steer = (1.0 if right_k else 0.0) - (1.0 if left_k else 0.0)
        factor = 0.55 + 0.45 * min(1.0, self.speed / 2.0)
        self.yaw += steer * config.TRAM_TURN * factor * dt

    # ---------------------------------------------------------------- физика
    def _integrate(self, dt):
        f = self.fwd()
        self.pos = self.pos + f * self.speed * dt
        self.speed = max(0.0, self.speed)    # страховка: назад ехать нельзя
        m = config.LOCO_WIDTH / 2 + 0.15
        if not (self.world.x0 + m <= self.pos[0] <= self.world.x1 - m) or \
           not (self.world.z0 + m <= self.pos[1] <= self.world.z1 - m):
            self.world.clamp_pos(self.pos, m)
            self.speed = 0.0
        for cr in self.world.crates:
            diff = self.pos - cr["pos"]
            d = math.hypot(diff[0], diff[1])
            rr = cr["radius"] + config.LOCO_WIDTH / 2 + 0.05
            if 1e-4 < d < rr:
                # тяжёлый состав отодвигает лёгкую тару меньше, но уверенно
                self.pos = cr["pos"] + diff / d * rr
                self.speed *= 0.4

    # ------------------------------------------------- наезды и задевания
    def _check_pedestrians(self):
        """Коллизия состава с работниками.

        Зона кузова: тягач + все вагоны → −25 (НАЕЗД).
        Зона выступа груза (шире кузова) → −20 (ГРУЗ УДАРИЛ РАБОТНИКА)."""
        if self.speed < 0.4:
            return
        segs = self.segments()
        cargo_segs = self.cargo_segments()
        for ped in self.world.pedestrians:
            if ped.knocked > 0:
                continue
            hit = False
            for (x0, z0, x1, z1, hw) in segs:
                d = point_seg_dist(ped.pos[0], ped.pos[1], x0, z0, x1, z1)
                if d < hw + config.PED_RADIUS:
                    self.speed *= 0.25       # резкое торможение при ударе
                    self.world.register_hit(ped, self.pos)
                    hit = True
                    break
            if not hit:
                for (x0, z0, x1, z1, hw) in cargo_segs:
                    d = point_seg_dist(ped.pos[0], ped.pos[1], x0, z0, x1, z1)
                    if d < hw + config.PED_RADIUS:
                        self.speed *= 0.5
                        self.world.register_cargo_hit(ped, self.pos)
                        break

    def _check_npc(self):
        """Столкновение с роботом II → −30."""
        npc = getattr(self.world, "npc", None)
        if npc is None:
            return
        for (x0, z0, x1, z1, hw) in self.segments():
            for (nx0, nz0, nx1, nz1, nhw) in npc.segments():
                # расстояние между отрезками ≈ через середины + полуразмеры
                d = point_seg_dist((nx0 + nx1) / 2, (nz0 + nz1) / 2,
                                   x0, z0, x1, z1)
                if d < hw + nhw:
                    self.world.register_npc_hit()
                    # расталкивание: состав игрока отходит от робота II
                    cx, cz = (nx0 + nx1) / 2, (nz0 + nz1) / 2
                    diff = self.pos - np.array([cx, cz])
                    dd = math.hypot(diff[0], diff[1]) or 1e-4
                    self.pos = self.pos + diff / dd * max(0.0, hw + nhw - dd)
                    self.speed *= 0.3
                    npc.brake(2.0)
                    return

    def _check_walls(self):
        """Кузов или ГРУЗ задел стену/стеллаж/рем.стол/барьер → штраф.

        - кузов: −15, выталкивание, скорость ×0.3;
        - груз (выступ шире кузова): −10, скорость ×0.75;
        - штраф не чаще WALL_COOLDOWN с на одно препятствие (кулдаун в мире),
          выталкивание выполняется каждый кадр контакта независимо от кулдауна;
        - ящики/тележки на полу штрафов не дают."""
        rects = self.world.solid_rects
        if not rects:
            return
        for cargo_mode in (False, True):
            segs = self.cargo_segments() if cargo_mode else self.segments()
            for si, (x0, z0, x1, z1, hw) in enumerate(segs):
                dx, dz = x1 - x0, z1 - z0
                seg_len = math.hypot(dx, dz)
                n = max(1, int(seg_len / 0.3))
                for (rx0, rz0, rx1, rz1, kind, key) in rects:
                    # быстрый отсев по расширенным AABB
                    if max(x0, x1) < rx0 - hw or min(x0, x1) > rx1 + hw or \
                       max(z0, z1) < rz0 - hw or min(z0, z1) > rz1 + hw:
                        continue
                    contact = False
                    px = pz = 0.0
                    for i in range(n + 1):
                        t = i / n
                        px = x0 + dx * t
                        pz = z0 + dz * t
                        if rx0 - hw <= px <= rx1 + hw and \
                           rz0 - hw <= pz <= rz1 + hw:
                            contact = True
                            break
                    if contact:
                        self.world.register_wall_hit(cargo_mode, kind, key)
                        if cargo_mode:
                            self.speed *= 0.75
                        else:
                            if si == 0:
                                # выталкивание только для тягача
                                dl = px - (rx0 - hw)
                                dr = (rx1 + hw) - px
                                db = pz - (rz0 - hw)
                                df = (rz1 + hw) - pz
                                mmp = min(dl, dr, db, df)
                                if mmp == dl:
                                    self.pos[0] -= dl
                                elif mmp == dr:
                                    self.pos[0] += dr
                                elif mmp == db:
                                    self.pos[1] -= db
                                else:
                                    self.pos[1] += df
                            self.speed *= 0.3
                        break

    # -------------------------------------------------------- след и вагоны
    def _place_wagons(self):
        """Вагоны встают на след тягача: интервалы gaps() по дуге."""
        n = self.n_wagons
        if not self.trail:
            return
        gaps = self.gaps()
        need = gaps[0]
        acc = 0.0
        wi = 0
        prev = None
        for p in reversed(self.trail):       # новые → старые
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
        if wi < n:                            # след короче состава — ставим вплотную
            last = self.wagon_pos[wi - 1] if wi > 0 else np.array(self.trail[0])
            lyaw = self.wagon_yaw[wi - 1] if wi > 0 else self.yaw
            while wi < n:
                self.wagon_pos[wi] = np.array([last[0], last[1]])
                self.wagon_yaw[wi] = lyaw
                wi += 1

    # ------------------------------------------------------------- отрисовка
    def dynamic_parts(self):
        """Список (mesh, pos3, yaw): тягач + вагоны (с грузом — слитно)."""
        out = [(self.m_loco, (self.pos[0], 0.0, self.pos[1]), self.yaw)]
        for i in range(self.n_wagons):
            wp = self.wagon_pos[i]
            wy = self.wagon_yaw[i]
            mesh = (self.m_wag_loaded[i % len(self.m_wag_loaded)]
                    if self.cargo and self.cargo_count > i
                    else self.m_wag[i % len(self.m_wag)])
            out.append((mesh, (wp[0], 0.0, wp[1]), wy))
        return out

    def draw(self, r):
        for mesh, pos, yaw in self.dynamic_parts():
            r.draw_mesh(mesh, pos, yaw)
        # сцепки между единицами состава
        pts = [(self.pos[0], self.pos[1])] + \
              [(wp[0], wp[1]) for wp in self.wagon_pos]
        segs = []
        for k in range(len(pts) - 1):
            segs.append((pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1]))
        r.draw_lines(segs, (30, 32, 38), width=2, y=0.30)

    # -------------------------------------------------- лучи лидаров в 3D
    def draw_sensors(self, r, lid_top=None, lid_bot=None):
        """Визуализация лидаров в главной 3D-сцене: развёртка верхнего,
        веер нижнего (прорежено ради FPS)."""
        if lid_top is not None and lid_top.scan:
            ox, _oy, oz = lid_top.origin
            segs = []
            for (ang, dist, _k) in lid_top.scan:
                dd = abs((ang - lid_top.angle + math.pi) % math.tau - math.pi)
                if dd > 0.35:                # узкий веер у развёртки
                    continue
                segs.append((ox, oz, ox + math.sin(ang) * dist,
                             oz + math.cos(ang) * dist))
            r.draw_lines(segs, config.LIDAR_TOP_COLOR, width=1,
                         y=lid_top.height)
        if lid_bot is not None and lid_bot.scan:
            ox, _oy, oz = lid_bot.origin
            segs = []
            for i, (ang, dist, _k) in enumerate(lid_bot.scan):
                if i % 3:
                    continue
                segs.append((ox, oz, ox + math.sin(ang) * dist,
                             oz + math.cos(ang) * dist))
            r.draw_lines(segs, config.LIDAR_BOTTOM_COLOR, width=1, y=0.06)
