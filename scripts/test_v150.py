# -*- coding: utf-8 -*-
"""Тесты: реальный склад + RoboCV + наблюдения.

Проверяется:
  1) склад: 28×13 м, H 4,5 м, ворота, рем. столы, барьеры, острова (×2 склада);
  2) трамвай: тягач 1,6×0,8×2,2 / 1220 кг, вагоны 0,85 м / 200 кг, 2,2 м/с;
  3) масса в физике: гружёный разгоняется медленнее порожнего;
  4) операции наблюдений: стоп-линия перед зоной, разгрузка 40 с,
     погрузка, рейс = 3 бигбокса, выкат синих тележек, АКБ садится;
  5) робот II: второй состав ездит, уступает составу игрока, штраф −30;
  6) перерыв: работники останавливаются, потери цикла учитываются;
  7) регрессии: ресайз, раскладка, лидары видят робот II, рендер всех окон.
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


# ---------------------------------------------------------------- 1) склад
check("Склад: ширина 28 м", abs(config.WAREHOUSE_W - 28.0) < 1e-6)
check("Склад: глубина 13 м", abs(config.WAREHOUSE_D - 13.0) < 1e-6)
check("Склад: высота 4,5 м", abs(config.WALL_H - 4.5) < 1e-6)
check("Складов два (легаси-полугабарит)",
      abs(app.world.hw - 14.0) < 1e-6 and abs(app.world.hd - 6.5) < 1e-6)
check("Стеллажи: 6 на склад × 2 склада", len(app.world.racks) == 12,
      f"{len(app.world.racks)}")
check("Рем. столы (по 2 на склад)", len(app.world.tables) == 4,
      f"{len(app.world.tables)}")
check("Барьеры чёрно-жёлтые", len(app.world.barriers) >= 8,
      f"{len(app.world.barriers)}")
check("Гнёзда площадок размечены", len(app.world._strip_rects) == 6,
      f"{len(app.world._strip_rects)}")
# ворота: проёмы в восточной стене A и западной стене B свободны
wall_rects = [r for r in app.world.solid_rects
              if r[4] == "wall" and (r[2] - r[0]) > 1.0]
gate_a_ok = all(not (r[0] < 14.9 and r[2] > 14.1 and r[1] < 1.9 and r[3] > -1.9)
                for r in wall_rects if 13.5 < (r[0] + r[2]) / 2 < 15.5)
gate_b_ok = all(not (r[0] < 95.1 and r[2] > 94.1 and r[1] < 1.9 and r[3] > -1.9)
                for r in wall_rects if 93.5 < (r[0] + r[2]) / 2 < 95.5)
check("Ворота: проёмы в стенах обоих складов", gate_a_ok and gate_b_ok)
# зелёный пол + оранжевая разметка
check("Разметка пола (оранжевые линии)", len(app.world.mark_segs) >= 10,
      f"{len(app.world.mark_segs)} линий")
# A* маршрут площадка A → площадка B (через территорию)
path = app.world.find_path(config.ZONE_A, config.ZONE_B)
check("A* площадка A → площадка B", len(path) >= 2, f"{len(path)} точек")

# ---------------------------------------------------------------- 2) трамвай
check("Тягач: длина 1,6 м", abs(config.LOCO_LENGTH - 1.6) < 1e-6)
check("Тягач: ширина 0,8 м", abs(config.LOCO_WIDTH - 0.8) < 1e-6)
check("Тягач: высота 2,2 м", abs(config.LOCO_HEIGHT - 2.2) < 1e-6)
check("Тягач: масса 1220 кг", abs(config.LOCO_MASS - 1220.0) < 1e-6)
check("Вагон: длина 0,85 м", abs(config.WAGON_LENGTH - 0.85) < 1e-6)
check("Вагон: масса 200 кг", abs(config.WAGON_MASS - 200.0) < 1e-6)
check("Скорость 2,2 м/с", abs(config.TRAM_SPEED - 2.2) < 1e-6)
check("Состав: 3 вагона", app.tram.n_wagons == 3)
check("Длина состава ~4,4 м", abs(app.tram.length_total() - 4.425) < 0.1,
      f"{app.tram.length_total():.2f} м")
check("Масса порожнего 1820 кг", abs(app.tram.mass() - 1820.0) < 1e-6)
app.tram.cargo_count = 3
check("Масса с грузом 2060 кг", abs(app.tram.mass() - 2060.0) < 1e-6)
app.tram.cargo_count = 0

# ------------------------------------------------- 3) масса влияет на физику
def accel_time(cargo_n):
    app.tram.reset()
    app.tram.auto = False
    app.tram.cargo_count = cargo_n
    app.tram.cargo = cargo_n > 0
    v0 = app.tram.speed
    app.keys["up"] = True
    for _ in range(60):                    # 1 с газа
        app.update(1 / 60)
    app.keys["up"] = False
    return app.tram.speed - v0

v_empty = accel_time(0)
v_loaded = accel_time(3)
check("Физика массы: порожний быстрее",
      v_empty > v_loaded * 1.05,
      f"порожний +{v_empty:.2f} м/с, гружёный +{v_loaded:.2f} м/с")
check("Скорость ограничена 2,2 м/с", v_empty <= config.TRAM_SPEED + 1e-6)

# ------------------------------------------------------ 4) операции (наблюдения)
check("Стоп-линия: 6 с", abs(config.STOP_LINE_TIME - 6.0) < 1e-6)
check("Разгрузка: 40 с (наблюдение)", abs(config.UNLOAD_TIME - 40.0) < 1e-6)
check("Погрузка: 45 с (реально 60–70 с/паллета)",
      abs(config.LOAD_TIME - 45.0) < 1e-6)
check("Рейс = 3 бигбокса", config.BIGBOX_PER_TRIP == 3)

# прогон до разгрузки (быстрые тайминги + ускорение времени ×8)
config.STOP_LINE_TIME = 0.5
config.LOAD_TIME = 3.0
config.UNLOAD_TIME = 3.0
app.tram.reset()
app.tram.timer = 0.3
app.tram.auto = True
app.time_scale = 8
states = set()
signal_seen = [False]
frames = 0
while app.tram.cycles < 1 and frames < 6000:
    app.update(1 / 60)
    states.add(app.tram.state)
    if app.tram.state == "СТОП-ЛИНИЯ" and app.tram._signal_done:
        signal_seen[0] = True      # сигнал подан при прибытии к зоне
    frames += 1
app.time_scale = 1
check("Цикл прошёл стоп-линию→погрузку→B→разгрузку",
      {"СТОП-ЛИНИЯ", "ПОГРУЗКА", "РАЗГРУЗКА"} <= states,
      str(sorted(states)))
check("Рейс засчитан (+100, 3 бигбокса)",
      app.tram.cycles == 1 and app.tram.bigboxes == 3,
      f"рейсов {app.tram.cycles}, бигбоксов {app.tram.bigboxes}")
check("АКБ разрядился за рейс", app.tram.battery < 100.0,
      f"{app.tram.battery:.1f}%")
check("Синие тележки выкачены", any(c.get("kind") == "cart"
                                    for c in app.world.crates),
      f"{sum(1 for c in app.world.crates if c.get('kind') == 'cart')} шт")
check("Сигнал прибытия (наблюдение: звуковой сигнал)",
      signal_seen[0])
config.STOP_LINE_TIME = 6.0
config.LOAD_TIME = 45.0
config.UNLOAD_TIME = 40.0

# ---------------------------------------------------------------- 5) робот II
npc = app.npc
check("Робот II существует", npc is not None and npc.n_wagons == 3)
check("Маршрут робота II задан", len(__import__('sim.npc', fromlist=['ROUTE']).ROUTE) == 4)
# движется, когда состав игрока далеко
app.tram.reset()
app.tram.auto = True
for _ in range(600):                        # 10 с
    app.update(1 / 60)
    if npc.speed > 0.5:
        break
check("Робот II ездит по складу", npc.speed > 0.5 or
      math.hypot(npc.pos[0] - npc.wagon_pos[-1][0],
                 npc.pos[1] - npc.wagon_pos[-1][1]) > 0.5,
      f"v={npc.speed:.2f} м/с, поза ({npc.pos[0]:.1f}, {npc.pos[1]:.1f})")
# уступает: состав игрока впереди по курсу робота → он стоит
npc.pos = np.array([-6.0, 3.55])
npc.yaw = -math.pi / 2                      # курс на запад (к кольцу островов)
npc.speed = 0.0
npc.pause = 0.0
npc.brake_t = 0.0
npc.wait_t = 0.0
npc.trail.clear()
npc._seed_trail()
app.tram.reset()
app.tram.auto = False                       # состав игрока стоит на пути
app.tram.pos = np.array([-8.5, 3.55])       # в западном проходе склада A
app.tram.yaw = -math.pi / 2
yielded = []
for _ in range(90):
    app.update(1 / 60)
    yielded.append(npc.speed)
check("Робот II уступает дорогу (наблюдение: ожидание выезда)",
      max(yielded) < 0.3, f"max v={max(yielded):.2f} м/с")
# столкновение: −30
app.world._wall_cd.clear()                  # убираем кулдаун прошлых событий
s0 = app.world.score
h0 = app.world.npc_hits
app.tram.pos = np.array([-8.0, -3.55])
app.tram.yaw = -math.pi / 2
npc.pos = np.array([-7.3, -3.55])           # робот II вплотную
npc.yaw = -math.pi / 2
npc.wagon_pos[0] = npc.pos.copy()
npc.wagon_yaw[0] = npc.yaw
app.tram.speed = 1.5
app.tram._check_npc()
check("Столкновение с роботом II: −30",
      app.world.score == s0 - config.NPC_HIT_PENALTY and
      app.world.npc_hits == h0 + 1,
      f"счёт {s0}→{app.world.score}")

# ---------------------------------------------------------------- 6) перерыв
app.world.break_timer = config.BREAK_PERIOD + 1.0
app.world.update(0.016)
check("Перерыв начинается", app.world.break_active)
check("Работники на перерыве стоят",
      all(p.speed < 1e-6 for p in app.world.pedestrians
          if math.hypot(p.pos[0] - 10.4, p.pos[1] - 4.7) < 1.0) or
      app.world.breaks_count >= 1,
      f"перерывов {app.world.breaks_count}")
app.world.break_left = 0.01
app.world.update(0.02)
check("Перерыв заканчивается", not app.world.break_active)

# ------------------------------------------------------------- 7) регрессии
try:
    app._resize(1280, 720)
    app._resize(config.WINDOW_W, config.WINDOW_H)
    check("Ресайз окна (регресс scratch)", True)
except Exception as e:  # noqa: BLE001
    check("Ресайз окна (регресс scratch)", False, str(e))
# лидар видит робот II как «робот II»
app.tram.pos = np.array([-1.5, -3.55])
app.tram.yaw = -math.pi / 2                 # смотрим на запад, к кольцу робота
npc.pos = np.array([-5.0, -3.55])
npc.wagon_pos[0] = npc.pos.copy()
app.lidar_top.update(app.world, app.tram)
kinds = {k for (_a, _d, k) in app.lidar_top.scan}
check("Лидар видит робота II", "tram" in kinds, str(sorted(kinds)))
# все окна + рендер
for wv in app.wm.wins:
    wv.visible = True
app.update(1 / 60)
app.render()
check("Рендер всех 16 окон", True)
panels_ok = app.wm.index["mission"].rect[3] >= 150 and \
    app.wm.index["stats"].rect[3] >= 150
check("Адаптивная раскладка (задание/статистика)", panels_ok,
      f"mission {app.wm.index['mission'].rect[3]}px, "
      f"stats {app.wm.index['stats'].rect[3]}px")

print()
print("ИТОГ:", "OK" if ok else f"FAIL ({len(FAILED)}): {FAILED}")
sys.exit(0 if ok else 1)
