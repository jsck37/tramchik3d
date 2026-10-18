# -*- coding: utf-8 -*-
"""Мир «Трамвайчик 3D»: ДВА одинаковых склада + маршрут 150 м.

Планировка (по чертежу, 28 × 13 × 4,5 м каждый):
  - СКЛАД A (центр 0,0, ПОГРУЗКА): ворота в ВОСТОЧНОЙ стене (проём 4 м);
  - СКЛАД B (центр 109,0, ВЫГРУЗКА): зеркальная копия, ворота в ЗАПАДНОЙ стене;
  - зоны погрузки/выгрузки у ворот — БЕЗ стеллажей: обычный зелёный пол,
    оранжевая площадка с гнёздами под груз и МАЛЫЙ КРУГ РАЗВОРОТА;
    груз со площадки убирают работники на штабелёрах (наблюдение);
  - стеллажи — вдоль дальних стен и 2 острова в западной/восточной половине;
  - МАРШРУТ A→B: 30 + 34,5 + 21 + 34,5 + 30 = 150 м, 4 поворота, коридор
    огорожен чёрно-жёлтыми барьерами, разметка — оранжевый пунктир;
  - пол ЗЕЛЁНЫЙ с оранжевой разметкой (наблюдение), на территории — бетон;
  - синие тележки выкатываются при разгрузке, мусор у площадки B;
  - обеденный перерыв: работники уходят к своим рем. столам (наблюдение).
"""
import math
import random
import numpy as np

import config
from .nav import NavGrid
from .pedestrians import Pedestrian
from .stacker import Stacker
from engine.mesh import box_mesh, quad_mesh, cylinder_mesh, merge_meshes, merge_parts

PAD = config.PAL

# Малый круг разворота: точки U-образного разворота (въезд по оси z=0,
# разворот через смещение +Z, выезд по той же оси в обратном направлении).
LOOP_A_WPTS = [(8.6, 0.0, 0.55), (7.9, 1.3, 0.5), (6.9, 2.2, 0.5),
               (5.4, 2.6, 0.5), (3.7, 2.3, 0.5), (2.6, 1.3, 0.5),
               (2.2, 0.0, 0.55)]
LOOP_B_WPTS = [(100.4, 0.0, 0.55), (101.1, 1.3, 0.5), (102.1, 2.2, 0.5),
               (103.6, 2.6, 0.5), (105.3, 2.3, 0.5), (106.4, 1.3, 0.5),
               (106.8, 0.0, 0.55)]


class World:
    def __init__(self):
        self.x0, self.z0, self.x1, self.z1 = config.WORLD_RECT
        self.cx_a = config.WH_A_CX
        self.cx_b = config.WH_B_CX
        self.rng = random.Random(777)        # склад
        self.time = 0.0
        self.zones = {"A": config.ZONE_A, "B": config.ZONE_B}
        # легаси-поля (часть тестов/кодa): полугабарит ОДНОГО склада
        self.hw = config.WAREHOUSE_W / 2.0
        self.hd = config.WAREHOUSE_D / 2.0
        # --- счёт и события (наезды/стены/доставки) ---
        self.score = 0
        self.hits = 0            # наезды кузовом
        self.wall_hits = 0       # кузов задел стену/стеллаж
        self.cargo_wall_hits = 0 # груз задел стену/стеллаж
        self.cargo_ped_hits = 0  # груз ударил работника
        self.npc_hits = 0        # столкновения с роботом II
        self.events = []          # [(текст, цвет)] → тосты в UI
        self._wall_cd = {}        # (по_cargo, key) → время штрафа (кулдаун)
        self.solid_rects = []     # жёсткие препятствия: (x0,z0,x1,z1,kind,key)
        self.lidar_walls = []     # AABB стен (для лидара)
        self._barrier_parts = []  # части барьеров — отдельные меши

        # --- наблюдения: перерыв / мусор / тележки ---
        self.break_timer = 0.0
        self.break_active = False
        self.break_left = 0.0
        self.breaks_count = 0
        self.break_loss = 0       # суммарные потери цикла, сек (+300 за перерыв)
        self.trash_timer = 0.0
        self.carts = []           # синие тележки: dict(pos, ttl, mesh)

        # --- навигация (прямоугольная сетка всего мира) ---
        self.nav = NavGrid(self.x0, self.z0, self.x1, self.z1, config.CELL)

        # --- статическая геометрия (меш строится один раз) ---
        self.static_parts = []   # (mesh, (x,y,z), yaw)
        self.grid_segs = []
        self.mark_segs = []      # оранжевая разметка пола
        self.zone_border_segs = []
        self.crates = []         # для коллизий/восприятия: dict(pos, size, ...)
        self.racks = []
        self.tables = []         # рем. столы (для лидара/карты)
        self.barriers = []       # чёрно-жёлтые барьеры
        self._strip_rects = []   # гнёзда площадок (для мини-карты)
        self.stackers = []       # штабелёры
        self._build_static()
        self._build_nav()
        # слияние статики в ПРОСТРАНСТВЕННЫЕ ЧАНКИ: каждый чанк — отдельный
        # меш, целиком отбрасывается по bbox за камерой/вне экрана;
        # барьеры/ограждения — ОТДЕЛЬНЫЕ меши (bbox = линия барьера →
        # целиком отсеиваются дёшево, не раздувая чанки)
        CHX, CHZ = 28.0, 13.0
        groups = {}
        for mesh, pos, yaw in self.static_parts:
            vs = mesh.vlist
            if vs:
                n = len(vs)
                mx = sum(v[0] for v in vs) / n
                mz = sum(v[2] for v in vs) / n
                ca, sa = math.cos(yaw), math.sin(yaw)
                wx = mx * ca + mz * sa + pos[0]
                wz = -mx * sa + mz * ca + pos[2]
            else:
                wx, wz = pos[0], pos[2]
            key = (int((wx - self.x0) // CHX), int((wz - self.z0) // CHZ))
            groups.setdefault(key, []).append((mesh, pos, yaw))
        self.static_meshes = [merge_meshes(groups[k]) for k in sorted(groups)]
        self.static_parts = []
        # отдельные меши барьеров
        for parts in self._barrier_parts:
            if parts:
                self.static_meshes.append(merge_meshes(parts))

        # --- маршруты миссии (малый круг → ворота → 150 м коридор) ---
        self._paths = {
            "to_b": ([(x, z, c) for (x, z, c) in LOOP_A_WPTS] +
                     [(5.0, 0.0, 0.8), (14.0, 0.0, 0.85),
                      (44.0, 0.0, 0.8), (44.0, -34.5, 0.55),
                      (65.0, -34.5, 0.55), (65.0, 0.0, 0.55),
                      (95.0, 0.0, 0.85), (98.4, 0.0, 0.7)]),
            "to_a": ([(x, z, c) for (x, z, c) in LOOP_B_WPTS] +
                     [(104.0, 0.0, 0.8), (95.0, 0.0, 0.85),
                      (65.0, 0.0, 0.8), (65.0, -34.5, 0.55),
                      (44.0, -34.5, 0.55), (44.0, 0.0, 0.55),
                      (14.0, 0.0, 0.85), (10.6, 0.0, 0.7)]),
        }

        # --- пешеходы ---
        self.pedestrians = []
        self._spawn_pedestrians()

        # --- штабелёры: убирают груз с площадок (наблюдение) ---
        self.stacker_a = Stacker(self, "A")
        self.stacker_b = Stacker(self, "B")
        self.stackers = [self.stacker_a, self.stacker_b]

        # --- шаблоны мешей пешехода: 8 фаз шага СЛИТЫ в один меш каждая ---
        c = PAD["ped"][0]
        torso = box_mesh(0.52, 0.62, 0.30, c)
        head = box_mesh(0.26, 0.28, 0.26, (232, 196, 160))
        limb = box_mesh(0.15, 0.86, 0.18, c)
        arm = box_mesh(0.12, 0.60, 0.15, c)
        self._ped_walk = []
        for i in range(8):
            swing = math.sin(i * math.tau / 8.0) * 0.26
            self._ped_walk.append(merge_parts([
                (torso, (0.0, 1.18, 0.0), 0.0),
                (head, (0.0, 1.62, 0.0), 0.0),
                (limb, (0.15, 0.45, swing), 0.0),
                (limb, (-0.15, 0.45, -swing), 0.0),
                (arm, (0.34, 1.12, -swing * 0.7), 0.0),
                (arm, (-0.34, 1.12, swing * 0.7), 0.0),
            ]))
        self._ped_down = merge_parts([
            (box_mesh(0.52, 0.28, 0.64, c), (0.0, 0.20, 0.0), 0.0),
            (head, (0.0, 0.24, 0.55), 0.0),
            (box_mesh(0.15, 0.18, 0.86, c), (0.16, 0.12, -0.50), 0.0),
            (box_mesh(0.15, 0.18, 0.86, c), (-0.16, 0.12, -0.50), 0.0),
            (box_mesh(0.12, 0.15, 0.60, c), (0.34, 0.10, 0.25), 0.0),
            (box_mesh(0.12, 0.15, 0.60, c), (-0.34, 0.10, 0.25), 0.0),
        ])
        # шаблон синей тележки (выкатывается при разгрузке)
        self._cart_mesh = merge_meshes([
            (box_mesh(0.62, 0.08, 0.80, (40, 44, 52), 0, 0.10, 0), (0, 0, 0), 0.0),
            (box_mesh(0.58, 0.34, 0.74, PAD["cart"], 0, 0.32, 0), (0, 0, 0), 0.0),
            (box_mesh(0.66, 0.06, 0.06, (30, 34, 40), 0, 0.62, -0.42), (0, 0, 0), 0.0),
        ])
        # динамические ящики (после разгрузки / мусор)
        self._dyn_meshes = {
            "bigbox": box_mesh(0.78, 0.62, 0.70, PAD["bigbox"], 0, 0.31, 0),
            "trash": box_mesh(0.45, 0.35, 0.45, PAD["trash"], 0, 0.18, 0),
        }

    # ================================================================ геом.
    def contains(self, x, z, m=0.0):
        return (self.x0 + m <= x <= self.x1 - m and
                self.z0 + m <= z <= self.z1 - m)

    def clamp_pos(self, pos, margin):
        """Ограничить позицию (np.array [x, z]) границами мира."""
        pos[0] = min(max(pos[0], self.x0 + margin), self.x1 - margin)
        pos[1] = min(max(pos[1], self.z0 + margin), self.z1 - margin)

    def mission_path(self, name):
        return list(self._paths[name])

    # ================================================================ меш мира
    def _build_nav(self):
        # стены-периметр складов и барьеры — в сетку A*
        for (x0, z0, x1, z1) in self.lidar_walls:
            self.nav.block_rect(x0, z0, x1, z1,
                                (self.zones["A"], self.zones["B"]), 2.6)
        for (x0, z0, x1, z1, kind, key) in self.solid_rects:
            if kind in ("rack", "table", "barrier"):
                self.nav.block_rect(x0 - 0.25, z0 - 0.25, x1 + 0.25, z1 + 0.25,
                                    (self.zones["A"], self.zones["B"]), 2.6)

    def _add_crate(self, x, z, sx, sy, sz, color, kind="crate", dyn=False):
        kind_ru = {"bigbox": "бигбокс", "pallet": "поддон", "crate": "ящик",
                   "cart": "тележка", "trash": "мусор"}.get(kind, "ящик")
        self.crates.append({"pos": np.array([x, z]), "size": (sx, sy, sz),
                            "radius": max(sx, sz) * 0.5 + 0.22, "color": color,
                            "kind": kind, "kind_ru": kind_ru, "ttl": None,
                            "dyn": dyn})
        if not dyn:
            self.static_parts.append(
                (box_mesh(sx, sy, sz, color, 0, sy * 0.5, 0), (x, 0.0, z), 0.0))

    def _add_rack(self, x, z, sx, sz, bays=None):
        """Стеллаж: стойки по периметру + 2 полки + груз. Возвращает AABB."""
        col = PAD["rack"]
        h = 3.4
        bays = bays or max(2, int(max(sx, sz) / 1.5))
        n_px = max(2, int(sx / 3.2) + 1)
        n_pz = max(2, int(sz / 3.2) + 1)
        for k in range(n_px):
            px = -sx / 2 + k * sx / (n_px - 1)
            for pz in (-sz / 2 + 0.10, sz / 2 - 0.10):
                self.static_parts.append(
                    (box_mesh(0.14, h, 0.14, col, px, h / 2, pz), (x, 0, z), 0.0))
        for k in range(n_pz):
            pz = -sz / 2 + k * sz / (n_pz - 1)
            for px in (-sx / 2 + 0.10, sx / 2 - 0.10):
                self.static_parts.append(
                    (box_mesh(0.14, h, 0.14, col, px, h / 2, pz), (x, 0, z), 0.0))
        for y in (1.15, 2.35):
            self.static_parts.append(
                (box_mesh(sx, 0.10, sz, col, 0, y, 0), (x, 0, z), 0.0))
        # груз на полках (декор): бигбоксы/ящики
        rnd = self.rng
        nb = max(1, int(sx * sz / 6.0))
        for i in range(nb):
            bx = rnd.uniform(-sx / 2 + 0.5, sx / 2 - 0.5)
            bz = rnd.uniform(-sz / 2 + 0.4, sz / 2 - 0.4)
            y = rnd.choice((1.2, 2.4))
            bw = rnd.uniform(0.55, 0.85)
            bc = PAD["bigbox"] if rnd.random() < 0.55 else PAD["crate"][i % 4]
            self.static_parts.append(
                (box_mesh(bw, 0.55, bw * 0.9, bc, bx, y + 0.28, bz), (x, 0, z), 0.0))
        self.racks.append({"pos": np.array([x, z]), "size": (sx, sz)})
        key = "rack%d" % (len(self.racks) - 1)
        self.solid_rects.append((x - sx / 2, z - sz / 2, x + sx / 2, z + sz / 2,
                                 "rack", key))
        return (x - sx / 2, z - sz / 2, x + sx / 2, z + sz / 2)

    def _add_table(self, x, z, sx, sz, key):
        """Ремонтный стол: столешница + ножки + тиски (наблюдение: «рем. стол»)."""
        col = (150, 154, 166)
        self.static_parts.append(
            (box_mesh(sx, 0.12, sz, col, 0, 0.86, 0), (x, 0, z), 0.0))
        for px in (-sx / 2 + 0.10, sx / 2 - 0.10):
            for pz in (-sz / 2 + 0.10, sz / 2 - 0.10):
                self.static_parts.append(
                    (box_mesh(0.10, 0.86, 0.10, (70, 74, 84), px, 0.43, pz),
                     (x, 0, z), 0.0))
        self.static_parts.append(
            (box_mesh(0.34, 0.20, 0.22, (90, 96, 110), -sx * 0.22, 1.02, 0),
             (x, 0, z), 0.0))
        self.static_parts.append(
            (box_mesh(0.28, 0.16, 0.30, PAD["bigbox"], sx * 0.24, 1.0, 0.05),
             (x, 0, z), 0.0))
        self.tables.append({"pos": np.array([x, z]), "size": (sx, sz)})
        self.solid_rects.append(
            (x - sx / 2, z - sz / 2, x + sx / 2, z + sz / 2, "table", key))

    def _add_barrier(self, x0, z0, x1, z1, key, kind="barrier"):
        """Чёрно-жёлтый барьер (наблюдение: опасные зоны огорожены)."""
        dx, dz = x1 - x0, z1 - z0
        L = math.hypot(dx, dz)
        if L < 0.05:
            return
        n = max(1, int(L / 1.1))
        ux, uz = (dx / L, dz / L) if L > 1e-6 else (1.0, 0.0)
        parts = []
        for i in range(n):
            t = (i + 0.5) / n
            px = x0 + dx * t
            pz = z0 + dz * t
            col = PAD["barrier_b"] if i % 2 == 0 else PAD["barrier_a"]
            parts.append((box_mesh(abs(ux) * L / n + abs(uz) * 0.12,
                                   0.30, abs(uz) * L / n + abs(ux) * 0.12, col,
                                   0, 0.15, 0), (px, 0, pz), 0.0))
            parts.append((box_mesh(0.08, 0.55, 0.08, PAD["barrier_a"],
                                   0, 0.28, 0), (px, 0, pz), 0.0))
        self._barrier_parts.append(parts)
        self.barriers.append({"a": (x0, z0), "b": (x1, z1)})
        self.solid_rects.append(
            (min(x0, x1) - 0.1, min(z0, z1) - 0.1, max(x0, x1) + 0.1,
             max(z0, z1) + 0.1, kind, key))

    def _add_gate(self, cx, z_wall, half, key, top):
        """Ворота в СЕВЕРНОЙ стене склада (z=z_wall): проём x∈[cx-half, cx+half].
        Зеркала (въезд) или знаки (выезд) — по наблюдениям."""
        col = PAD["pole"]
        h = config.WALL_H
        for gx in (cx - half + 0.15, cx + half - 0.15):
            self.static_parts.append(
                (box_mesh(0.3, h, 0.3, col, 0, h / 2, 0), (gx, 0, z_wall), 0.0))
            self.solid_rects.append(
                (gx - 0.15, z_wall - 0.15, gx + 0.15, z_wall + 0.15,
                 "wall", key))
        self.static_parts.append(
            (box_mesh(2 * half - 0.3, 0.3, 0.24, (90, 96, 112), cx, h - 0.4,
                      z_wall), (0, 0, 0), 0.0))
        if top:
            for gx in (cx - half + 0.15, cx + half - 0.15):
                self.static_parts.append(
                    (cylinder_mesh(0.28, 0.06, 10, (210, 224, 240), axis="z",
                                   cy=2.6, cz=0), (gx, 0, z_wall), 0.0))
        else:
            for gx, col_s in ((cx - half + 0.15, (90, 200, 110)),
                              (cx + half - 0.15, (230, 80, 70))):
                self.static_parts.append(
                    (box_mesh(0.56, 0.7, 0.08, col_s, 0, 2.6, 0),
                     (gx, 0, z_wall), 0.0))
                self.static_parts.append(
                    (box_mesh(0.6, 0.08, 0.1, (240, 240, 240), 0, 2.42, 0),
                     (gx, 0, z_wall), 0.0))

    # ------------------------------------------------------------- один склад
    def _build_warehouse(self, cx, mirror, tag):
        """Один склад: 28×13×4,5 м. mirror=False → ворота ВОСТОК (склад A),
        mirror=True → ворота ЗАПАД (склад B, зеркально)."""
        hw, hd = config.WAREHOUSE_W / 2.0, config.WAREHOUSE_D / 2.0
        x0, x1 = cx - hw, cx + hw
        z0, z1 = -hd, hd
        gx = x1 if not mirror else x0          # x ворот
        padx = config.ZONE_A[0] if not mirror else config.ZONE_B[0]
        m = -1.0 if mirror else 1.0            # множитель отражения по X

        def X(local):                          # локальный x → мировой
            return cx + m * local

        wall_col = PAD["wall"]
        H = config.WALL_H
        # пол — ЗЕЛЁНЫЙ (наблюдение: «пол зеленый с оранжевыми линиями»)
        self.static_parts.append(
            (quad_mesh((x0, 0, z0), (x0, 0, z1), (x1, 0, z1), (x1, 0, z0),
                       PAD["floor"]), (0, 0, 0), 0.0))
        # --- стены с проёмом ворот 4 м (на восточной/западной стене) ---
        g0, g1 = -config.GATE_HALF, config.GATE_HALF   # проём z∈[-2,2]
        # северная и южная стены — целиком (длиннее склада на угол)
        all_walls = [
            (cx, z0 - 0.5, config.WAREHOUSE_W + 2.0, 1.0),     # север
            (cx, z1 + 0.5, config.WAREHOUSE_W + 2.0, 1.0),     # юг
        ]
        if not mirror:
            # склад A: проём в ВОСТОЧНОЙ стене, западная — целиком
            all_walls.append((x0 - 0.5, 0.0, 1.0, config.WAREHOUSE_D))
            all_walls += [((x1 + 0.5), (z0 + g0) / 2.0, 1.0, (g0 - z0)),
                          ((x1 + 0.5), (z1 + g1) / 2.0, 1.0, (z1 - g1))]
        else:
            # склад B: проём в ЗАПАДНОЙ стене, восточная — целиком
            all_walls += [((x0 - 0.5), (z0 + g0) / 2.0, 1.0, (g0 - z0)),
                          ((x0 - 0.5), (z1 + g1) / 2.0, 1.0, (z1 - g1)),
                          ((x1 + 0.5), 0.0, 1.0, config.WAREHOUSE_D)]
        for wi, (wx, wz, sx, sz) in enumerate(all_walls):
            self.static_parts.append(
                (box_mesh(sx, H, sz, wall_col, 0, H / 2, 0), (wx, 0, wz), 0.0))
            self.solid_rects.append(
                (wx - sx / 2, wz - sz / 2, wx + sx / 2, wz + sz / 2,
                 "wall", "%s_wall%d" % (tag, wi)))
            self.lidar_walls.append(
                (wx - sx / 2, wz - sz / 2, wx + sx / 2, wz + sz / 2))
        # ворота со стойками/зеркалами/знаками — в стене x=gx
        self._add_gate_v(gx, g0, g1, "%s_gate" % tag, top=not mirror)

        # ---- стеллажи ВДОЛЬ ДАЛЬНИХ СТЕН (западная половина) ----
        # зона погрузки/выгрузки у ворот (ближняя треть) — БЕЗ стеллажей!
        for lz, lsize in ((-5.65, 1.4), (5.65, 1.4)):
            for lx, lsx in ((-9.5, 7.0), (-2.0, 5.0)):
                r = self._add_rack(X(lx), lz, lsx, lsize)
                self.solid_rects.append(
                    (r[0] - 0.25, r[1] - 0.25, r[2] + 0.25, r[3] + 0.25,
                     "rack", "%s_r%d" % (tag, len(self.racks))))
        # два стеллажных острова (чертёж: два прямоугольника в центре)
        for li, (lx, lsx, bays) in enumerate(((-7.5, 4.5, 3), (-1.5, 2.0, 2))):
            r = self._add_rack(X(lx), 0.0, lsx, 4.8, bays=bays)

        # ---- рем. столы + барьеры (западная половина, наблюдение) ----
        self._add_table(X(-11.5), -4.9, 2.0, 0.85, "%s_t1" % tag)
        self._add_barrier(X(-12.6), -3.95, X(-10.4), -3.95, "%s_b1" % tag)
        self._add_table(X(-11.5), 4.9, 2.0, 0.85, "%s_t2" % tag)
        self._add_barrier(X(-12.6), 3.95, X(-10.4), 3.95, "%s_b2" % tag)

        # ---- зарядная станция (наблюдение: «зарядка АКБ у ворот») ----
        czx = X(-4.0)
        self.static_parts.append(
            (box_mesh(0.9, 1.5, 0.5, (86, 96, 120), 0, 0.75, 0),
             (czx, 0, z1 - 0.35), 0.0))
        self.static_parts.append(
            (box_mesh(0.5, 0.3, 0.1, (90, 220, 130), 0, 1.2, -0.27),
             (czx, 0, z1 - 0.35), 0.0))

        # ---- ручная кладка у дальней стены (наблюдение) ----
        self.static_parts.append(
            (box_mesh(3.0, 0.09, 0.9, (128, 132, 146), 0, 0.045, 0),
             (X(-12.0), 0, z1 - 0.45), 0.0))
        for i in range(4):
            self._add_crate(X(-13.3 + i * 0.72), z1 - 0.42 - (i % 2) * 0.18,
                            0.52, 0.4, 0.5, PAD["crate"][i % 4])

        # ---- тара в карманах у стен (западная половина) ----
        pockets = [(-12.9, -2.6), (-12.85, 2.6), (-5.2, -5.8), (-0.6, -5.85),
                   (-5.2, 5.8), (-0.6, 5.85), (-13.0, -4.6), (-13.0, 4.6)]
        for i, (lx, lz) in enumerate(pockets):
            r = self.rng.random()
            if r < 0.45:
                sx, sy, col, k = 0.8, 0.62, PAD["bigbox"], "bigbox"
            elif r < 0.7:
                sx, sy, col, k = 0.85, 0.16, PAD["pallet"], "pallet"
            else:
                sx, sy, col, k = 0.62, 0.5, PAD["crate"][i % 4], "crate"
            if sy < 0.3:      # поддон — низкий, штабель из 2
                self._add_crate(X(lx), lz, sx, sy, sx * 0.7, col, k)
                self._add_crate(X(lx) + 0.05, lz, sx * 0.9, 0.4, sx * 0.65,
                                PAD["crate"][(i + 1) % 4], "crate")
            else:
                self._add_crate(X(lx), lz, sx, sy, sx * 0.75, col, k)

        # ---- фонарные столбы по углам ----
        for sx_ in (-1, 1):
            for sz_ in (-1, 1):
                x, z = cx + sx_ * (hw - 0.9), sz_ * (hd - 0.9)
                self.static_parts.append(
                    (box_mesh(0.18, H, 0.18, PAD["pole"], 0, H / 2, 0),
                     (x, 0, z), 0.0))
                self.static_parts.append(
                    (box_mesh(0.5, 0.2, 0.5, PAD["lamp"], 0, H + 0.1, 0),
                     (x, 0, z), 0.0))

        # ---- разметка: сетка пола (шаг 3 м — экономия линий) ----
        step = 3.0
        for v in np.arange(x0, x1 + 0.001, step):
            self.grid_segs.append((float(v), z0, float(v), z1))
        for v in np.arange(z0, z1 + 0.001, step):
            self.grid_segs.append((x0, float(v), x1, float(v)))

        mm = self.mark_segs
        # площадка ПОГРУЗКИ/ВЫГРУЗКИ (оранжевая рамка + гнёзда)
        zw, zd = config.ZONE_SIZE
        px0, pz0 = padx - zw / 2, -zd / 2
        px1, pz1 = padx + zw / 2, zd / 2
        self.static_parts.append(
            (quad_mesh((px0, 0.04, pz0), (px0, 0.04, pz1), (px1, 0.04, pz1),
                       (px1, 0.04, pz0), PAD["pad_a"] if not mirror
                       else PAD["pad_b"], no_cull=False), (0, 0, 0), 0.0))
        for (ax, az, bx, bz) in ((px0, pz0, px1, pz0), (px1, pz0, px1, pz1),
                                 (px1, pz1, px0, pz1), (px0, pz1, px0, pz0)):
            self.zone_border_segs.append((ax, az, bx, bz))
        # гнёзда под 3 бигбокса (север площадки — там их ставят/забирают)
        for k in (-1, 0, 1):
            sx0 = padx + k * 1.05 - 0.45
            self._strip_rects.append((sx0, -2.35, sx0 + 0.9, -1.45))
            mm.append((sx0, -2.35, sx0 + 0.9, -2.35))
            mm.append((sx0, -1.45, sx0 + 0.9, -1.45))
            mm.append((sx0, -2.35, sx0, -1.45))
            mm.append((sx0 + 0.9, -2.35, sx0 + 0.9, -1.45))
        # СТОП-ЛИНИЯ перед площадкой (наблюдение: остановка 18–28 с)
        slx = padx + m * 2.2
        mm.append((slx, -2.0, slx, 2.0))
        # МАЛЫЙ КРУГ РАЗВОРОТА: окружность + центр + шевроныы
        lcx, lcz = (config.LOOP_A_C if not mirror else config.LOOP_B_C)
        R = config.LOOP_R
        n = 26
        for i in range(n):
            a0 = math.tau * i / n
            a1 = math.tau * (i + 1) / n
            mm.append((lcx + math.cos(a0) * R, lcz + math.sin(a0) * R,
                       lcx + math.cos(a1) * R, lcz + math.sin(a1) * R))
        mm.append((lcx - 0.18, lcz - 0.18, lcx + 0.18, lcz + 0.18))
        mm.append((lcx - 0.18, lcz + 0.18, lcx + 0.18, lcz - 0.18))
        for k in range(3):                    # шевроны направления разворота
            aa = math.tau * (0.12 + 0.30 * k)
            bx_, bz_ = lcx + math.cos(aa) * R, lcz + math.sin(aa) * R
            tx, tz = -math.sin(aa), math.cos(aa)
            mm.append((bx_ - tx * 0.45 - math.cos(aa) * 0.3,
                       bz_ - tz * 0.45 - math.sin(aa) * 0.3,
                       bx_, bz_))
            mm.append((bx_, bz_, bx_ + tx * 0.45 - math.cos(aa) * 0.3,
                       bz_ + tz * 0.45 - math.sin(aa) * 0.3))

    def _add_gate_v(self, gx, gz0, gz1, key, top):
        """Ворота в вертикальной (восточной/западной) стене: проём z∈[gz0, gz1]."""
        col = PAD["pole"]
        h = config.WALL_H
        for gz in (gz0 + 0.15, gz1 - 0.15):
            self.static_parts.append(
                (box_mesh(0.3, h, 0.3, col, 0, h / 2, 0), (gx, 0, gz), 0.0))
            self.solid_rects.append(
                (gx - 0.15, gz - 0.15, gx + 0.15, gz + 0.15, "wall", key))
        self.static_parts.append(
            (box_mesh(0.24, 0.3, gz1 - gz0 - 0.3, (90, 96, 112), 0, h - 0.4,
                      (gz0 + gz1) / 2), (gx, 0, 0), 0.0))
        if top:   # зеркала на въезде (наблюдение)
            for gz in (gz0 + 0.15, gz1 - 0.15):
                self.static_parts.append(
                    (cylinder_mesh(0.28, 0.06, 10, (210, 224, 240), axis="x",
                                   cy=2.6, cx=0), (gx, 0, gz), 0.0))
        else:     # знаки на выезде
            for gz, col_s in ((gz0 + 0.15, (90, 200, 110)),
                              (gz1 - 0.15, (230, 80, 70))):
                self.static_parts.append(
                    (box_mesh(0.08, 0.7, 0.56, col_s, 0, 2.6, 0),
                     (gx, 0, gz), 0.0))
                self.static_parts.append(
                    (box_mesh(0.1, 0.08, 0.6, (240, 240, 240), 0, 2.42, 0),
                     (gx, 0, gz), 0.0))

    # ------------------------------------------------------------- территория
    def _offset_polyline(self, pts, d):
        """Сдвиг ломаной на ±d (митры под 90°). Возвращает (лево, право)."""
        n = len(pts)
        left, right = [], []
        for i in range(n):
            dirs = []
            if i > 0:
                dx, dz = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
                L = math.hypot(dx, dz) or 1.0
                dirs.append((dx / L, dz / L))
            if i < n - 1:
                dx, dz = pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]
                L = math.hypot(dx, dz) or 1.0
                dirs.append((dx / L, dz / L))
            if len(dirs) == 1:
                dx, dz = dirs[0]
                nl, nr = (-dz, dx), (dz, -dx)
                left.append((pts[i][0] + nl[0] * d, pts[i][1] + nl[1] * d))
                right.append((pts[i][0] + nr[0] * d, pts[i][1] + nr[1] * d))
            else:
                (ax, az), (bx, bz) = dirs           # направления сегментов
                nl1, nr1 = (-az, ax), (az, -ax)     # нормали сегмента 1
                nl2, nr2 = (-bz, bx), (bz, -bx)     # нормали сегмента 2
                # митра: пересечение сдвинутых ПРЯМЫХ СЕГМЕНТОВ
                for (out, n1, n2, d1, d2) in (
                        (left, nl1, nl2, (ax, az), (bx, bz)),
                        (right, nr1, nr2, (ax, az), (bx, bz))):
                    p1 = (pts[i][0] + n1[0] * d, pts[i][1] + n1[1] * d)
                    p2 = (pts[i][0] + n2[0] * d, pts[i][1] + n2[1] * d)
                    det = d1[0] * d2[1] - d1[1] * d2[0]
                    if abs(det) < 1e-6:
                        out.append(((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))
                    else:
                        # p1 + t*d1 = p2 + s*d2
                        rx_, rz_ = p2[0] - p1[0], p2[1] - p1[1]
                        t = (rx_ * d2[1] - rz_ * d2[0]) / det
                        out.append((p1[0] + d1[0] * t, p1[1] + d1[1] * t))
        return left, right

    def _build_corridor(self):
        """Территория между складами: бетон, ограждения, разметка 150 м."""
        pts = config.ROUTE_WPTS
        d = config.ROUTE_HALF_W
        left, right = self._offset_polyline(pts, d)
        # бетонное покрытие коридора (полосы по сегментам)
        for i in range(len(pts) - 1):
            (ax, az), (bx, bz) = pts[i], pts[i + 1]
            if abs(bz - az) < 1e-6:      # горизонтальный сегмент
                self.static_parts.append(
                    (quad_mesh((ax, 0, az - d), (ax, 0, az + d),
                               (bx, 0, az + d), (bx, 0, az - d),
                               (58, 62, 72)), (0, 0, 0), 0.0))
            else:                        # вертикальный
                zmin, zmax = min(az, bz), max(az, bz)
                self.static_parts.append(
                    (quad_mesh((ax - d, 0, zmin), (ax - d, 0, zmax),
                               (ax + d, 0, zmax), (ax + d, 0, zmin),
                               (58, 62, 72)), (0, 0, 0), 0.0))
        # тёмный грунт под всей территорией (фон за ограждениями)
        self.static_parts.append(
            (quad_mesh((self.x0 - 40, -0.02, self.z0 - 40),
                       (self.x0 - 40, -0.02, self.z1 + 40),
                       (self.x1 + 40, -0.02, self.z1 + 40),
                       (self.x1 + 40, -0.02, self.z0 - 40),
                       (34, 38, 46), no_cull=True), (0, 0, 0), 0.0))
        # ОГРАЖДЕНИЯ по обеим сторонам маршрута (чёрно-жёлтые барьеры)
        for side, poly in (("l", left), ("r", right)):
            for i in range(len(poly) - 1):
                (ax, az), (bx, bz) = poly[i], poly[i + 1]
                self._add_barrier(ax, az, bx, bz,
                                  "fence_%s%d" % (side, i), kind="fence")
        # воронки у ворот складов (от проёма 4 м к коридору 5,6 м)
        self._add_barrier(14.0, 2.0, 16.9, d, "funnel_a1", kind="fence")
        self._add_barrier(14.0, -2.0, 16.9, -d, "funnel_a2", kind="fence")
        self._add_barrier(95.0, 2.0, 92.1, d, "funnel_b1", kind="fence")
        self._add_barrier(95.0, -2.0, 92.1, -d, "funnel_b2", kind="fence")
        # разметка: оранжевый пунктир по центру + шевроны на поворотах
        mm = self.mark_segs
        dash = 2.2
        for i in range(len(pts) - 1):
            (ax, az), (bx, bz) = pts[i], pts[i + 1]
            L = math.hypot(bx - ax, bz - az)
            ux, uz = (bx - ax) / L, (bz - az) / L
            s = 1.0
            while s < L - 1.0:
                e = min(s + dash, L - 0.4)
                mm.append((ax + ux * s, az + uz * s, ax + ux * e, az + uz * e))
                s += 2 * dash
        # шевроны перед каждым поворотом (4 поворота — наблюдение/маршрут)
        for ti in range(1, len(config.ROUTE_WPTS) - 1):
            tx, tz = config.ROUTE_WPTS[ti]
            # входящее направление (откуда едем) и исходящее (куда поворачиваем)
            ix = tx - config.ROUTE_WPTS[ti - 1][0]
            iz = tz - config.ROUTE_WPTS[ti - 1][1]
            Li = math.hypot(ix, iz) or 1.0
            ix, iz = ix / Li, iz / Li
            ox = config.ROUTE_WPTS[ti + 1][0] - tx
            oz = config.ROUTE_WPTS[ti + 1][1] - tz
            Lo = math.hypot(ox, oz) or 1.0
            ox, oz = ox / Lo, oz / Lo
            for k in range(3):
                off = 2.0 + k * 1.2
                px_, pz_ = tx - ix * off, tz - iz * off
                pxv, pzv = -oz, ox          # перпендикуляр исходящего
                mm.append((px_ - ox * 0.5 - pxv * 0.55,
                           pz_ - oz * 0.5 - pzv * 0.55, px_, pz_))
                mm.append((px_, pz_, px_ - ox * 0.5 + pxv * 0.55,
                           pz_ - oz * 0.5 + pzv * 0.55))
        # вехи маршрута: столбики на каждых 30 м (30/60/90/120 м)
        acc = 0.0
        next_m = 30.0
        for i in range(len(pts) - 1):
            (ax, az), (bx, bz) = pts[i], pts[i + 1]
            L = math.hypot(bx - ax, bz - az)
            ux, uz = (bx - ax) / L, (bz - az) / L
            while next_m <= acc + L and next_m < config.ROUTE_LEN - 1:
                dd = next_m - acc
                mx_, mz_ = ax + ux * dd, az + uz * dd
                self.static_parts.append(
                    (box_mesh(0.12, 1.1, 0.12, PAD["pole"], 0, 0.55, 0),
                     (mx_ + uz * 1.9, 0, mz_ - ux * 1.9), 0.0))
                self.static_parts.append(
                    (box_mesh(0.5, 0.36, 0.06, (250, 210, 70), 0, 1.0, 0),
                     (mx_ + uz * 1.9, 0, mz_ - ux * 1.9), 0.0))
                next_m += 30.0
            acc += L
        # фонари вдоль территории
        for (fx, fz) in ((22, 3.4), (44, 3.4), (66, -37.9), (88, 3.4)):
            self.static_parts.append(
                (box_mesh(0.18, 5.2, 0.18, PAD["pole"], 0, 2.6, 0),
                 (fx, 0, fz), 0.0))
            self.static_parts.append(
                (box_mesh(0.55, 0.2, 0.55, PAD["lamp"], 0, 5.3, 0),
                 (fx, 0, fz), 0.0))

    def _build_static(self):
        self._build_corridor()
        self._build_warehouse(self.cx_a, mirror=False, tag="A")
        self._build_warehouse(self.cx_b, mirror=True, tag="B")
        # запас груза у стеллажей склада A (оттуда штабелёр везёт на площадку)
        for k, zoff in enumerate((1.2, 0.0, -1.2)):
            self._add_crate(-4.8, zoff, 0.78, 0.62, 0.7, PAD["bigbox"], "bigbox")

    def _spawn_pedestrians(self):
        # (роам-прямоугольник, точка перерыва, стартовые точки)
        areas = [
            ((-12.0, -5.2, 12.5, 5.2), (-11.5, -3.6),
             [(-8.5, 2.5), (-3.5, -3.2)]),                    # склад A
            ((96.5, -5.2, 121.0, 5.2), (120.5, -3.6),
             [(100.5, 2.5), (116.0, -3.2)]),                  # склад B
            ((18.0, -30.0, 40.0, -4.0), (30.0, -20.0),
             [(25.0, -12.0), (34.0, -24.0)]),                 # территория
        ]
        n = 0
        for (roam, bspot, starts) in areas:
            for (sx, sz) in starts:
                if n >= config.N_PEDESTRIANS:
                    return
                ped = Pedestrian(self, sx, sz, PAD["ped"][n % len(PAD["ped"])])
                ped.roam = roam
                ped.break_spot = np.array(bspot, dtype=np.float64)
                ped.pos = np.array([sx, sz], dtype=np.float64)
                self.pedestrians.append(ped)
                n += 1

    # ================================================================ запросы
    def point_free(self, x, z, r):
        for cr in self.crates:
            if math.hypot(x - cr["pos"][0], z - cr["pos"][1]) < cr["radius"] + r:
                return False
        for rk in self.racks:
            if abs(x - rk["pos"][0]) < rk["size"][0] / 2 + r and \
               abs(z - rk["pos"][1]) < rk["size"][1] / 2 + r:
                return False
        for tb in self.tables:
            if abs(x - tb["pos"][0]) < tb["size"][0] / 2 + r and \
               abs(z - tb["pos"][1]) < tb["size"][1] / 2 + r:
                return False
        # барьеры/ограждения: не выбирать цели внутри/за барьерами
        for (bx0, bz0, bx1, bz1, kind, _key) in self.solid_rects:
            if kind not in ("barrier", "table"):
                continue
            if bx0 - r < x < bx1 + r and bz0 - r < z < bz1 + r:
                return False
        return True

    def find_path(self, p0, p1):
        return self.nav.find_path(p0, p1)

    # ================================================== наблюдения: перерыв
    def _start_break(self):
        self.break_active = True
        self.break_left = config.BREAK_DURATION
        self.breaks_count += 1
        tr = getattr(self, "tram", None)
        if tr is not None and getattr(tr, "state", "") == "РАЗГРУЗКА":
            self.break_loss += config.BREAK_CYCLE_LOSS
            self.events.append(
                (f"ПЕРЕРЫВ: разгрузка прервана, цикл +{config.BREAK_CYCLE_LOSS // 60} мин",
                 (255, 200, 80)))
        else:
            self.events.append(("ОБЕДЕННЫЙ ПЕРЕРЫВ — работники отошли",
                                (255, 200, 80)))

    def _spawn_trash(self):
        """Мусор у площадки разгрузки (наблюдение: «скопление мусора»)."""
        bx, bz = self.zones["B"]
        for _ in range(12):
            x = bx + self.rng.uniform(-2.6, 2.6)
            z = bz + self.rng.uniform(-2.6, -0.9)
            if self.point_free(x, z, 0.7):
                self.crates.append(
                    {"pos": np.array([x, z]), "size": (0.45, 0.35, 0.45),
                     "radius": 0.42, "color": PAD["trash"], "kind": "trash",
                     "kind_ru": "мусор", "ttl": 70.0, "dyn": True})
                self.events.append(("МУСОР У ПЛОЩАДКИ B — манёвры затруднены",
                                    (200, 190, 170)))
                return

    def on_unloaded(self):
        """После разгрузки: 3 бигбокса на площадке B — работники уберут их
        штабелёром (наблюдение: «груз ставят на пол, оттуда убирают»)."""
        bx, bz = self.zones["B"]
        for k in (-1, 0, 1):
            self.crates.append(
                {"pos": np.array([bx + k * 1.05, 1.9]),
                 "size": (0.78, 0.62, 0.7), "radius": 0.61,
                 "color": PAD["bigbox"], "kind": "bigbox",
                 "kind_ru": "бигбокс", "ttl": None, "dyn": True,
                 "staged": True})

    def spawn_carts(self, n=3):
        """Выкатить синие тележки у площадки разгрузки (наблюдение)."""
        bx, bz = self.zones["B"]
        spots = [(bx - 1.2, bz - 1.7), (bx - 2.3, bz - 1.7), (bx - 3.4, bz - 1.7)]
        for i in range(min(n, len(spots))):
            x, z = spots[i]
            if not self.point_free(x, z, 0.5):
                x, z = x + 0.5, z - 0.6
            cr = {"pos": np.array([x, z]), "size": (0.62, 0.55, 0.8),
                  "radius": 0.5, "color": PAD["cart"], "kind": "cart",
                  "kind_ru": "тележка", "ttl": 55.0, "dyn": True}
            self.crates.append(cr)
            self.carts.append({"pos": np.array([x, z]), "ttl": 55.0, "crate": cr})
        self.events.append(("СИНИЕ ТЕЛЕЖКИ ВЫКАЧЕНЫ К ПЛОЩАДКЕ B",
                            (120, 180, 255)))

    # ================================================================ счёт
    def add_score(self, delta, text, color):
        """Начислить/списать очки и отправить событие в UI."""
        self.score += delta
        self.events.append((text, color))

    def register_hit(self, ped, tram_pos):
        """Наезд на работника кузовом состава: падает, штраф, событие в UI."""
        if ped.knocked > 0:
            return
        ped.knock_down(tram_pos)
        self.hits += 1
        self.add_score(-config.HIT_PENALTY,
                       f"НАЕЗД НА РАБОТНИКА: −{config.HIT_PENALTY}",
                       PAD["warn"])

    def register_cargo_hit(self, ped, tram_pos):
        """Выступающий груз ударил работника: падает, штраф."""
        if ped.knocked > 0:
            return
        ped.knock_down(tram_pos)
        self.cargo_ped_hits += 1
        self.add_score(-config.CARGO_PED_PENALTY,
                       f"ГРУЗ УДАРИЛ РАБОТНИКА: −{config.CARGO_PED_PENALTY}",
                       (255, 170, 80))

    def register_npc_hit(self):
        """Столкновение состава с роботом II."""
        k = ("npc", "npc")
        if self.time - self._wall_cd.get(k, -99.0) < config.WALL_COOLDOWN:
            return False
        self._wall_cd[k] = self.time
        self.npc_hits += 1
        self.add_score(-config.NPC_HIT_PENALTY,
                       f"СТОЛКНОВЕНИЕ С РОБОТОМ II: −{config.NPC_HIT_PENALTY}",
                       (190, 150, 255))
        return True

    def register_wall_hit(self, by_cargo, kind, key):
        """Кузов или ГРУЗ задел препятствие. Штраф не чаще WALL_COOLDOWN."""
        k = (by_cargo, key)
        if self.time - self._wall_cd.get(k, -99.0) < config.WALL_COOLDOWN:
            return False
        self._wall_cd[k] = self.time
        if len(self._wall_cd) > 64:
            now = self.time
            self._wall_cd = {kk: tv for kk, tv in self._wall_cd.items()
                             if now - tv < 30.0}
        kind_ru = config.WALL_KIND_RU.get(kind, "ПРЕПЯТСТВИЕ")
        if by_cargo:
            self.cargo_wall_hits += 1
            self.add_score(-config.CARGO_WALL_PENALTY,
                           f"ГРУЗ ЗАДЕЛ {kind_ru}: −{config.CARGO_WALL_PENALTY}",
                           (255, 170, 80))
        else:
            self.wall_hits += 1
            self.add_score(-config.WALL_HIT_PENALTY,
                           f"ПОЕЗД ЗАДЕЛ {kind_ru}: −{config.WALL_HIT_PENALTY}",
                           PAD["warn"])
        return True

    def pop_events(self):
        ev = self.events
        self.events = []
        return ev

    # ================================================================ обновление
    def update(self, dt):
        self.time += dt
        for ped in self.pedestrians:
            ped.update(dt)
        for st in self.stackers:
            st.update(dt)
        # перерыв (наблюдение: во время перерыва цикл +5 мин)
        if self.break_active:
            self.break_left -= dt
            if self.break_left <= 0.0:
                self.break_active = False
                self.events.append(("ПЕРЕРЫВ ОКОНЧЕН — работники вернулись",
                                    (150, 220, 150)))
        else:
            self.break_timer += dt
            if self.break_timer >= config.BREAK_PERIOD:
                self.break_timer = 0.0
                self._start_break()
        # мусор у площадки B
        self.trash_timer += dt
        if self.trash_timer >= 90.0:
            self.trash_timer = 0.0
            if sum(1 for c in self.crates if c.get("kind") == "trash") < 2:
                self._spawn_trash()
        # старение мусора и тележек
        for cr in self.crates:
            if cr.get("ttl") is not None:
                cr["ttl"] -= dt
        self.crates = [c for c in self.crates if c.get("ttl") is None or c["ttl"] > 0]
        # тележки — это записи в crates (по kind), отдельный список для рендера
        self.carts = [{"pos": cr["pos"], "ttl": cr.get("ttl", 0.0), "crate": cr}
                      for cr in self.crates if cr.get("kind") == "cart"]

    # ================================================================= отрисовка
    def draw_scene(self, r, grid=True, peds=True):
        """Отправляет все меши мира в рендерер (сортировка произойдёт в end_view)."""
        if grid:
            r.draw_lines(self.grid_segs, PAD["grid"], width=1, y=0.02)
            r.draw_lines(self.mark_segs, PAD["mark"], width=2, y=0.03)
            r.draw_lines(self.zone_border_segs, PAD["mark"], width=2, y=0.06)
        # статика по чанкам (bbox-cull отбрасывает невидимые целиком)
        for sm in self.static_meshes:
            r.draw_mesh(sm, (0.0, 0.0, 0.0), 0.0)
        # динамические ящики (после разгрузки) и мусор
        for cr in self.crates:
            mesh = self._dyn_meshes.get(cr.get("kind")) if cr.get("dyn") else None
            if mesh is not None:
                r.draw_mesh(mesh, (cr["pos"][0], 0.0, cr["pos"][1]),
                            cr.get("yaw", 0.0))
        # синие тележки (динамика)
        for c in self.carts:
            r.draw_mesh(self._cart_mesh, (c["pos"][0], 0.0, c["pos"][1]),
                        c.get("yaw", 0.0))
        # штабелёры (v1.6.0)
        for st in self.stackers:
            st.draw(r)
        if peds:
            for ped in self.pedestrians:
                self._draw_pedestrian(r, ped)

    def _draw_pedestrian(self, r, ped):
        """Работник = ОДИН слитый меш фазы шага; сбитый — лежит.
        Пешеходы вне поля зрения камеры пропускаются целиком (culling)."""
        c = r.cam
        dx = ped.pos[0] - c.pos[0]
        dz = ped.pos[1] - c.pos[2]
        zv = dx * c.w[0] + dz * c.w[2]
        if zv < 1.0 or zv > 60.0:
            return
        xv = dx * c.u[0] + dz * c.u[2]
        if abs(xv) > zv * 1.0 + 3.0:
            return
        if ped.knocked > 0:
            r.draw_mesh(self._ped_down, (ped.pos[0], 0.0, ped.pos[1]), ped.yaw)
            return
        idx = int(ped.phase * 8.0 / math.tau) % 8
        r.draw_mesh(self._ped_walk[idx], (ped.pos[0], 0.0, ped.pos[1]), ped.yaw)


WAREHOUSE_W = config.WAREHOUSE_W   # легаси-имя (часть старых тестов)
