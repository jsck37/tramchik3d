# -*- coding: utf-8 -*-
"""Тесты: два склада, маршрут 150 м, малый круг, чистые зоны,
штабелёры, ускорение времени, вид от 1-го лица, объёмный рендер.

Проверяется:
  1) два одинаковых склада 28×13 (зеркальные), ворота на маршрут;
  2) маршрут 150 м, ровно 4 поворота, ограждения с корректными митрами;
  3) зоны погрузки/выгрузки БЕЗ стеллажей (обычный пол + площадки);
  4) малый круг разворота в обеих зонах + разметка;
  5) миссия: полный цикл A→B→A через малые круги (с ускорением ×8);
  6) штабелёры: забирают груз с площадки B, подвозят к площадке A;
  7) ускорение времени ×1/×2/×4/×8: симуляция идёт быстрее, рендер цел;
  8) вид от 1-го лица: камера в кабине, жёстко привязана к тягачу;
  9) регрессии: штрафы, лидары, ресайз, раскладка, рендер всех окон.
"""
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import main as m  # noqa: E402
import config  # noqa: E402


class A:
    selftest = True
    cam = None
    width = None
    height = None


app = m.App(A())
app.keys = {"up": False, "down": False, "left": False, "right": False,
            "brake": False}
ok = True
FAILED = []


def check(name, cond, info=""):
    global ok
    print(f"{name}: {'OK' if cond else 'FAIL'} {info}")
    if not cond:
        ok = False
        FAILED.append(name)


# ------------------------------------------------- 1) два одинаковых склада
w = app.world
check("Склады: центры A и B", config.WH_A_CX == 0.0 and config.WH_B_CX == 109.0)
wall_rects = [r for r in w.solid_rects if r[4] == "wall" and (r[2] - r[0]) > 1.0]
# у склада A проём в восточной стене (x=14..15, z∈[-2,2] свободен)
gate_a_free = all(not (r[0] < 14.9 and r[2] > 14.1 and r[1] < 1.9 and r[3] > -1.9)
                  for r in wall_rects if 13.5 < (r[0] + r[2]) / 2 < 15.5)
gate_b_free = all(not (r[0] < 95.1 and r[2] > 94.1 and r[1] < 1.9 and r[3] > -1.9)
                  for r in wall_rects if 93.5 < (r[0] + r[2]) / 2 < 95.5)
check("Ворота склада A (восток) свободны", gate_a_free)
check("Ворота склада B (запад) свободны", gate_b_free)
check("Стеллажей 12 (по 6 на склад)", len(w.racks) == 12, f"{len(w.racks)}")
# зеркальность: racks B = зеркальное отражение racks A
ax = sorted(round(r["pos"][0], 1) for r in w.racks if r["pos"][0] < 54.5)
bx = sorted(round(109.0 - r["pos"][0], 1) for r in w.racks if r["pos"][0] > 54.5)
check("Склад B зеркальная копия A", ax == bx, f"{len(ax)}/{len(bx)}")

# ------------------------------------- 2) маршрут 150 м с 4 поворотами
pts = config.ROUTE_WPTS
segs = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)]
check("Маршрут: 5 сегментов", len(segs) == 5, str([round(s, 1) for s in segs]))
check("Маршрут: длина 150 м", abs(sum(segs) - 150.0) < 1e-6,
      f"{sum(segs):.1f} м")
check("Маршрут: ровно 4 поворота", len(pts) - 2 == 4)
check("Маршрут: все сегменты прямые (ось X/Z)",
      all(abs(dx) < 1e-9 or abs(dz) < 1e-9
          for dx, dz in ((pts[i + 1][0] - pts[i][0],
                          pts[i + 1][1] - pts[i][1]) for i in range(4))))
fences = [r for r in w.solid_rects if r[4] == "fence"]
check("Ограждения маршрута установлены", len(fences) >= 12, f"{len(fences)}")
# ограждения не перекрывают осевую линию маршрута (полоса ±1.2 м свободна)
axis_free = True
for i in range(len(pts) - 1):
    (ax_, az_), (bx_, bz_) = pts[i], pts[i + 1]
    # прямоугольная проверка: рект пересекает полосу вдоль сегмента?
    for r in fences:
        x0, z0, x1, z1 = r[0], r[1], r[2], r[3]
        if abs(bz_ - az_) < 1e-6:      # горизонтальный сегмент
            if x1 > min(ax_, bx_) and x0 < max(ax_, bx_) and \
               z0 < az_ + 1.2 and z1 > az_ - 1.2:
                axis_free = False
        else:                          # вертикальный
            if z1 > min(az_, bz_) and z0 < max(az_, bz_) and \
               x0 < ax_ + 1.2 and x1 > ax_ - 1.2:
                axis_free = False
check("Ось маршрута свободна от ограждений", axis_free)

# --------------------------- 3) зоны погрузки/выгрузки БЕЗ стеллажей
def zone_clear(zx):
    """В прямоугольнике зоны (+малый круг) нет стеллажей/столов/БАРЬЕРОВ.
    (ограждения-воронки у ворот — часть коридора, не зоны; kind='fence')"""
    bad = []
    for rk in w.racks:
        rx, rz = rk["pos"]
        if abs(rx - zx) < 12.0 and abs(rz) < 4.0 and rx > zx - 9.5:
            bad.append(("rack", rx, rz))
    for tb in w.tables:
        if abs(tb["pos"][0] - zx) < 12.0 and abs(tb["pos"][1]) < 4.0:
            bad.append(("table",))
    for r in w.solid_rects:
        if r[4] != "barrier":
            continue
        bx_ = (r[0] + r[2]) / 2
        bz_ = (r[1] + r[3]) / 2
        if abs(bx_ - zx) < 12.0 and abs(bz_) < 3.0 and bx_ > zx - 9.5:
            bad.append(("barrier",))
    return bad

check("Зона A без стеллажей/столов/барьеров", not zone_clear(config.ZONE_A[0]),
      str(zone_clear(config.ZONE_A[0])[:3]))
check("Зона B без стеллажей/столов/барьеров", not zone_clear(config.ZONE_B[0]),
      str(zone_clear(config.ZONE_B[0])[:3]))

# ------------------------------------------------- 4) малый круг разворота
mm_near_a = sum(1 for (a, b, c, d) in w.mark_segs
                if math.hypot((a + c) / 2 - config.LOOP_A_C[0],
                              (b + d) / 2 - config.LOOP_A_C[1])
                < config.LOOP_R + 0.3)
mm_near_b = sum(1 for (a, b, c, d) in w.mark_segs
                if math.hypot((a + c) / 2 - config.LOOP_B_C[0],
                              (b + d) / 2 - config.LOOP_B_C[1])
                < config.LOOP_R + 0.3)
check("Малый круг A размечен", mm_near_a >= 20, f"{mm_near_a} линий")
check("Малый круг B размечен", mm_near_b >= 20, f"{mm_near_b} линий")
# точки разворота в маршрутах миссии
path_b = w.mission_path("to_b")
path_a = w.mission_path("to_a")
check("Маршрут to_b начинается малым кругом A",
      abs(path_b[0][0] - 8.6) < 0.1 and abs(path_b[0][1]) < 0.1)
check("Маршрут to_a начинается малым кругом B",
      abs(path_a[0][0] - 100.4) < 0.1 and abs(path_a[0][1]) < 0.1)
# разворот меняет направление на противоположное (въезд/выезд по z=0)
check("Круг A: въезд запад, выезд восток",
      path_b[0][0] > path_b[6][0] and path_b[6][0] < 3.0,
      f"въезд {path_b[0][0]} → выезд {path_b[6][0]}")
check("Круг B: въезд восток, выезд запад",
      path_a[0][0] < path_a[6][0],
      f"въезд {path_a[0][0]} → выезд {path_a[6][0]}")

# ------------------------------------------- 5) полный цикл миссии (×8)
config.STOP_LINE_TIME = 0.5
config.LOAD_TIME = 3.0
config.UNLOAD_TIME = 3.0
config.BREAK_PERIOD = 1e9
app.tram.reset()
app.tram.timer = 0.3
app.tram.auto = True
app.time_scale = 8
states = []
frames = 0
while app.tram.cycles < 1 and frames < 4000:
    app.update(1 / 60)
    if not states or states[-1] != app.tram.state:
        states.append(app.tram.state)
    frames += 1
app.time_scale = 1
check("Полный цикл за 4000 кадров (×8)", app.tram.cycles == 1,
      f"кадров {frames}, состояний {len(states)}")
check("Цикл: стоп-линия→погрузка→B→разгрузка→возврат",
      {"СТОП-ЛИНИЯ", "ПОГРУЗКА", "ДВИЖЕНИЕ К B", "РАЗГРУЗКА",
       "ВОЗВРАЩЕНИЕ К A"} <= set(states), str(states))
check("Рейс засчитан: +100, 3 бигбокса",
      app.tram.cycles == 1 and app.tram.bigboxes == 3 and
      app.world.score >= 100)
check("Без штрафов на чистом маршруте", app.world.wall_hits == 0 and
      app.world.hits == 0 and app.world.npc_hits == 0)
check("Груз выгружен на пол (штабелёр заберёт)",
      any(c.get("staged") for c in w.crates) or w.stacker_b._done > 0)
check("Синие тележки выкачены", any(c.get("kind") == "cart"
                                    for c in w.crates))

# ----------------------------------------------------- 6) штабелёры
st_a, st_b = w.stacker_a, w.stacker_b
check("Штабелёров два", len(w.stackers) == 2)
check("Штабелёр A поработал", st_a._done >= 1 or st_a.state != "IDLE",
      f"{st_a.state}, доставок {st_a._done}")
# принудительный прогон штабелёра B: груз на площадке → к стеллажу
w.crates.append({"pos": np.array([98.4, 1.9]), "size": (0.78, 0.62, 0.7),
                 "radius": 0.61, "color": (0, 0, 0), "kind": "bigbox",
                 "kind_ru": "бигбокс", "ttl": None, "dyn": True,
                 "staged": True})
n0 = st_b._done
n_staged0 = sum(1 for c in w.crates if c.get("staged"))
for _ in range(3200):
    st_b.update(1 / 60)
    if st_b._done > n0:
        break
check("Штабелёр B убрал груз с площадки", st_b._done > n0,
      f"состояние {st_b.state}, убрано {st_b._done - n0}")
check("Число грузов на площадке уменьшилось",
      sum(1 for c in w.crates if c.get("staged")) == n_staged0 - 1)

# ------------------------------------------------ 7) ускорение времени
app.tram.reset()
app.tram.auto = False
p0 = app.tram.pos.copy()
app.keys["up"] = True
app.time_scale = 8
for _ in range(60):
    app.update(1 / 60)
app.keys["up"] = False
app.time_scale = 1
d8 = float(np.hypot(*(app.tram.pos - p0)))
check("Ускорение ×8: проехано ~8× больше", d8 > 8.0,
      f"{d8:.1f} м за 60 кадров (1 с ×8 = 8 с сим.)")
check("Скорость не превышает максимум", app.tram.speed <= config.TRAM_SPEED + 1e-6)
scales = config.TIME_SCALES
check("Шкала ускорений ×1/×2/×4/×8", scales == [1, 2, 4, 8])
app._cycle_time_scale()
check("Переключение времени (T): ×1→×2", app.time_scale == 2)
app._cycle_time_scale(); app._cycle_time_scale(); app._cycle_time_scale()
check("Переключение по кругу: ×8→×1", app.time_scale == 1)

# ------------------------------------------------ 8) вид от 1-го лица
app.tram.reset()
app.view_mode = "orbit"
app._cycle_view()          # → chase
check("Вид: следящая после F", app.view_mode == "chase")
app._cycle_view()          # → cab
check("Вид: кабина после F×2", app.view_mode == "cab")
app.render()
cd = app.renderer.cam
t = app.tram
f = t.fwd()
expected_x = t.pos[0] + f[0] * config.CAB_VIEW_FWD
expected_z = t.pos[1] + f[1] * config.CAB_VIEW_FWD
check("Камера в кабине тягача", abs(cd.pos[0] - expected_x) < 0.01 and
      abs(cd.pos[2] - expected_z) < 0.01,
      f"{cd.pos[0]:.2f},{cd.pos[2]:.2f}")
check("Высота глаз в кабине", abs(cd.pos[1] - config.CAB_VIEW_HEIGHT) < 0.01)
check("Курс камеры = курс тягача",
      abs(cd.w[0] - f[0]) < 0.02 and abs(cd.w[2] - f[1]) < 0.02)
# жёсткая привязка: сдвигаем трамвай — камера следует
t.pos = t.pos + f * 3.0
app.render()
cd = app.renderer.cam          # камера пересоздаётся каждый кадр
check("Камера следует за тягачом (фиксированная)",
      abs(cd.pos[0] - (t.pos[0] + f[0] * config.CAB_VIEW_FWD)) < 0.01)
app._cycle_view()          # → orbit
check("Возврат к орбите", app.view_mode == "orbit" and app.cam.mode == "orbit")

# ------------------------------------------------------- 9) регрессии
try:
    app._resize(1280, 720)
    app._resize(config.WINDOW_W, config.WINDOW_H)
    check("Ресайз окна", True)
except Exception as e:  # noqa: BLE001
    check("Ресайз окна", False, str(e))
# лидар видит робота II (запад склада A)
app.tram.pos = np.array([-1.5, -3.55])
app.tram.yaw = -math.pi / 2
npc = app.npc
npc.pos = np.array([-5.0, -3.55])
npc.yaw = -math.pi / 2
npc.wagon_pos[0] = npc.pos.copy()
app.lidar_top.update(app.world, app.tram)
kinds = {k for (_a, _d, k) in app.lidar_top.scan}
check("Лидар видит робота II", "tram" in kinds, str(sorted(kinds)))
# штраф за стену работает (северная стена склада A)
app.tram.reset()
app.tram.auto = False
app.tram.pos = np.array([8.0, config.WORLD_HD - 4.0])
app.tram.yaw = 0.0
s0, wh0 = app.world.score, app.world.wall_hits
app.keys["up"] = True
for _ in range(300):
    app.update(1 / 60)
app.keys["up"] = False
check("Штраф за стену: −15 (регресс v1.3)",
      app.world.wall_hits > wh0 and app.world.score < s0)
# рендер всех окон + адаптивная сетка
for wv in app.wm.wins:
    wv.visible = True
app._toggle("m2")   # toggle + relayout
app.wm.index["m2"].visible = True
panels_ok = (app.wm.index["mission"].rect[3] >= 150 and
             app.wm.index["stats"].rect[3] >= 150 and
             app.wm.index["minimap"].rect[3] >= 140)
check("Адаптивная раскладка (карта/задание/статистика)", panels_ok,
      f"map {app.wm.index['minimap'].rect[3]}, mission "
      f"{app.wm.index['mission'].rect[3]}, stats {app.wm.index['stats'].rect[3]}")
try:
    app.update(1 / 60)
    app.render()
    check("Рендер всех 16 окон", True)
except Exception as e:  # noqa: BLE001
    check("Рендер всех 16 окон", False, str(e))
# одно открытое окно YOLO — компактная центрированная панель (не на весь экран)
for wv in app.wm.wins:
    wv.visible = False
app.wm.index["m1"].visible = True
app.wm.index["minimap"].visible = True
app.wm.index["mission"].visible = True
app.wm.index["stats"].visible = True
app.wm.index["lidar_top"].visible = True
app._toggle("m1")
app._toggle("m1")

import pygame  # noqa: E402
rr = pygame.Rect(app.wm.index["m1"].rect)
lt_right = (app.wm.index["lidar_top"].rect[0] +
            app.wm.index["lidar_top"].rect[2])
check("Одно окно YOLO: компактная панель в центре",
      rr.width <= 710 and rr.width > 300 and
      rr.centerx > lt_right and rr.centerx < app.wm.index["minimap"].rect[0],
      f"{rr.width}×{rr.height} @ {rr.centerx}")

print()
print("ИТОГ:", "OK" if ok else f"FAIL ({len(FAILED)}): {FAILED}")
sys.exit(0 if ok else 1)
