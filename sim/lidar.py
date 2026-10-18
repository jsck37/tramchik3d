# -*- coding: utf-8 -*-
"""Лидары «Трамвайчика 3D»: эмуляция двух сканеров.

- ВЕРХНИЙ — на крыше кабины паровозика, полный оборот 360°: обнаруживает
  людей, ящики, стеллажи и стены вокруг состава;
- НИЖНИЙ — в низу переднего бампера, сектор 120° вперёд: сканирует пол,
  ноги пешеходов и низкие препятствия (борюры, ящики).

Рейкаст в 2D (вид сверху): луч против окружностей (пешеходы, ящики) и
прямоугольников (стеллажи, стены). Результаты: scan = [(angle, dist, kind)],
фронтальная дистанция для автопилота, вращающаяся развёртка для радара.
"""
import math

import config

KIND_PERSON = "person"
KIND_CRATE = "crate"
KIND_RACK = "rack"
KIND_WALL = "wall"
KIND_TRAM = "tram"
KIND_NONE = "none"


def _ray_circle(ox, oz, dx, dz, cx, cz, r):
    """Ближайшее положительное t пересечения луча с окружностью или None."""
    mx, mz = cx - ox, cz - oz
    b = mx * dx + mz * dz
    if b < 0.0:                       # окружность позади луча
        return None
    disc = b * b - (mx * mx + mz * mz - r * r)
    if disc < 0.0:
        return None
    s = math.sqrt(disc)
    t1 = b - s
    if t1 > 1e-4:
        return t1
    t2 = b + s
    return t2 if t2 > 1e-4 else None


def _ray_aabb(ox, oz, dx, dz, x0, z0, x1, z1):
    """Слэб-метод: ближайшее t пересечения луча с AABB или None."""
    tmin, tmax = 0.0, 1e18
    for o, d, a, b in ((ox, dx, x0, x1), (oz, dz, z0, z1)):
        if abs(d) < 1e-9:
            if o < a or o > b:
                return None
            continue
        t1 = (a - o) / d
        t2 = (b - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return None
    return tmin if tmin > 1e-4 else None


class Lidar:
    """Однолучевой сканер с настраиваемым сектором; хватает препятствия мира."""

    def __init__(self, cfg, label, fov_offset_deg=0.0):
        self.label = label
        self.height = cfg["height"]
        self.dx = cfg["dx"]
        self.dz = cfg["dz"]
        self.fov = math.radians(cfg["fov"])
        self.rays = cfg["rays"]
        self.range = cfg["range_m"]
        self.spin_dps = cfg["spin_dps"]
        self.every = max(1, cfg.get("every", 2))
        self.full_circle = self.fov >= math.tau - 1e-6
        self.fov_offset = math.radians(fov_offset_deg)
        self.angle = 0.0            # текущий угол развёртки (визуализация)
        self.scan = []              # [(angle_world, dist, kind)]
        self.n_scans = 0
        self.origin = (0.0, 0.0, 0.0)   # мировая точка подвеса (x, y, z)
        self.frame = 0

    # ------------------------------------------------------------- геометрия
    def pose(self, tram):
        """Точка подвеса и базовый угол (мировые координаты)."""
        s, c = math.sin(tram.yaw), math.cos(tram.yaw)
        lx, lz = self.dx, self.dz
        ox = tram.pos[0] + lx * c + lz * s
        oz = tram.pos[1] - lx * s + lz * c
        return ox, oz, tram.yaw + self.fov_offset

    # ------------------------------------------------------------- сканирование
    def update(self, world, tram):
        ox, oz, base = self.pose(tram)
        self.origin = (ox, self.height, oz)
        self.angle = base if not self.full_circle else \
            (self.angle + math.radians(self.spin_dps) * (1.0 / 30.0)) % math.tau
        hits = []
        if self.full_circle:
            step = math.tau / self.rays
            angles = [base + i * step for i in range(self.rays)]
        else:
            half = self.fov * 0.5
            angles = [base - half + self.fov * i / (self.rays - 1)
                      for i in range(self.rays)]
        obstacles = self._collect(world, ox, oz)
        for ang in angles:
            dx, dz = math.sin(ang), math.cos(ang)
            best_t = self.range
            kind = KIND_NONE
            for (typ, o) in obstacles:
                if typ in (KIND_PERSON, KIND_CRATE, KIND_TRAM):
                    t = _ray_circle(ox, oz, dx, dz, o[0], o[1], o[2])
                else:
                    t = _ray_aabb(ox, oz, dx, dz, o[0], o[1], o[2], o[3])
                if t is not None and t < best_t:
                    best_t, kind = t, typ
            hits.append((ang % math.tau, best_t, kind))
        self.scan = hits
        self.n_scans += 1
        return hits

    def _collect(self, world, ox=0.0, oz=0.0):
        """Сбор препятствий. prefilter по дальности — препятствия
        дальше range от лидара не проверяются лучами (мир стал 137×45 м)."""
        R = self.range
        out = []
        for ped in world.pedestrians:
            if ped.knocked > 0:
                r = 0.75                      # лежащий — шире, ниже
            else:
                r = 0.38
            if abs(ped.pos[0] - ox) < R + 1.0 and abs(ped.pos[1] - oz) < R + 1.0:
                out.append((KIND_PERSON, (ped.pos[0], ped.pos[1], r)))
        for cr in world.crates:
            if abs(cr["pos"][0] - ox) < R + 1.0 and abs(cr["pos"][1] - oz) < R + 1.0:
                out.append((KIND_CRATE, (cr["pos"][0], cr["pos"][1],
                                         cr["radius"] * 0.9)))
        for rk in world.racks:
            x, z = rk["pos"]
            sx, sz = rk["size"]
            if abs(x - ox) < R + sx and abs(z - oz) < R + sz:
                out.append((KIND_RACK, (x - sx / 2, z - sz / 2,
                                        x + sx / 2, z + sz / 2)))
        for tb in getattr(world, "tables", []):
            x, z = tb["pos"]
            sx, sz = tb["size"]
            if abs(x - ox) < R + sx and abs(z - oz) < R + sz:
                out.append((KIND_RACK, (x - sx / 2, z - sz / 2,
                                        x + sx / 2, z + sz / 2)))
        # штабелёры (v1.6.0)
        for st in getattr(world, "stackers", []):
            if abs(st.pos[0] - ox) < R + 1.0 and abs(st.pos[1] - oz) < R + 1.0:
                out.append((KIND_CRATE, (st.pos[0], st.pos[1], 0.55)))
        # робот II — второй состав (v1.5.0): круги по единицам состава
        npc = getattr(world, "npc", None)
        if npc is not None:
            if abs(npc.pos[0] - ox) < R + 3.0 and abs(npc.pos[1] - oz) < R + 3.0:
                out.append((KIND_TRAM, (npc.pos[0], npc.pos[1], 0.62)))
                for wp in npc.wagon_pos:
                    out.append((KIND_TRAM, (wp[0], wp[1], 0.55)))
        # стены обоих складов (v1.6.0: список из мира) — только близкие
        for (x0, z0, x1, z1) in getattr(world, "lidar_walls", []):
            if x1 < ox - R or x0 > ox + R or z1 < oz - R or z0 > oz + R:
                continue
            out.append((KIND_WALL, (x0, z0, x1, z1)))
        # ограждения маршрута — низкие: видит только НИЖНИЙ лидар
        if self.height < 1.2:
            for (x0, z0, x1, z1, kind, _key) in world.solid_rects:
                if kind not in ("barrier", "fence"):
                    continue
                if x1 < ox - R or x0 > ox + R or z1 < oz - R or z0 > oz + R:
                    continue
                out.append((KIND_WALL, (x0, z0, x1, z1)))
        return out

    # ------------------------------------------------------------- запросы
    def front_clearance(self, tram, sector_deg=35.0):
        """Минимальная дистанция в секторе вперёд по курсу (для автопилота)."""
        if not self.scan:
            return self.range
        half = math.radians(sector_deg)
        best = self.range
        for (ang, dist, _kind) in self.scan:
            d = abs((ang - tram.yaw + math.pi) % math.tau - math.pi)
            if d <= half and dist < best:
                best = dist
        return best

    def nearest(self):
        """(dist, kind) ближайшего отражения в последнем скане."""
        if not self.scan:
            return self.range, KIND_NONE
        dist, kind = min(((d, k) for (_a, d, k) in self.scan), key=lambda x: x[0])
        return dist, kind

    def hits_by_kind(self):
        """Счётчик отражений по типам (для панелей)."""
        cnt = {k: 0 for k in (KIND_PERSON, KIND_CRATE, KIND_RACK, KIND_WALL,
                              KIND_TRAM)}
        for (_a, _d, k) in self.scan:
            if k in cnt:
                cnt[k] += 1
        return cnt
