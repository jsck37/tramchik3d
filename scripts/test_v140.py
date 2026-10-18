# -*- coding: utf-8 -*-
"""Тесты: штрафные очки — поезд или ГРУЗ задел стену/стеллаж/работника.

Проверяется:
  1) кузов о стену: −15, выталкивание, скорость гасится;
  2) груз о стену: груз выступает за габарит и штрафуется (−10) раньше кузова;
  3) классификация работник: кузов → −25 (НАЕЗД), выступ груза → −20 (ГРУЗ);
  4) кулдаун: одно препятствие не штрафуется каждый кадр;
  5) регресс: наезд кузовом по-прежнему −25, доставка +100 не задета.
"""
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
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
# быстрые операции (проверяем штрафы, не хронометраж)
config.STOP_LINE_TIME = 1.0
config.LOAD_TIME = 4.0
config.UNLOAD_TIME = 4.0
config.BREAK_PERIOD = 1e9
app.tram.timer = min(app.tram.timer, 1.0)
ok = True


def drive(frames):
    for _ in range(frames):
        app.update(1 / 60)


def place(x, z, yaw=0.0, cargo=False):
    app.tram.reset()
    app.tram.auto = False
    app.tram.pos = np.array([x, z], dtype=np.float64)
    app.tram.yaw = yaw
    app.tram.cargo = cargo
    app.tram.cargo_count = 3 if cargo else 0   # груз виден по счётчику
    app.tram.trail.clear()
    app.tram._seed_trail()


# 1) кузов задел стену: −15, выталкивание внутрь склада (стена севера z=+hd,
#    x=8 — восточнее ворот, там стена сплошная)
place(8.0, config.WORLD_HD - 4.0, 0.0, cargo=False)   # едем к северной стене
s0, wh0 = app.world.score, app.world.wall_hits
app.keys["up"] = True
drive(300)
app.keys["up"] = False
walls_hit = app.world.wall_hits - wh0
inside = app.tram.pos[1] < config.WORLD_HD - 0.2
print(f"1) Кузов→стена: штрафов {walls_hit}, счёт {s0} → {app.world.score}, "
      f"z={app.tram.pos[1]:.2f} (внутрь выталкивает: {inside})")
ok &= walls_hit >= 1
ok &= app.world.score <= s0 - config.WALL_HIT_PENALTY
ok &= inside

# 2) ГРУЗ задел стену: выступ шире кузова — груз-штраф БЕЗ касания кузова
#    (вагон ставим так, чтобы зону стены пересекал только выступ груза:
#     кузов hw 0.35 → касание от hd-0.35; груз hw 0.41 → от hd-0.41)
app.tram.reset()
app.tram.auto = False
app.tram.cargo = True
app.tram.cargo_count = 3
app.tram.speed = 1.0
app.tram.wagon_pos[1] = np.array([8.0, config.WORLD_HD - 0.79])  # вагон 2
app.tram.wagon_yaw[1] = 0.0
s0, cw0, wh0 = (app.world.score, app.world.cargo_wall_hits,
                app.world.wall_hits)
app.tram._check_walls()
cargo_w = app.world.cargo_wall_hits - cw0
body_w = app.world.wall_hits - wh0
print(f"2) Груз→стена: груз-штрафов {cargo_w} (−{config.CARGO_WALL_PENALTY}), "
      f"кузов-штрафов {body_w}, счёт {s0} → {app.world.score}")
ok &= cargo_w == 1 and body_w == 0
ok &= app.world.score == s0 - config.CARGO_WALL_PENALTY
# 2б) при более глубоком контакте штрафуются и груз, и кузов (разные события)
app.world._wall_cd.clear()
app.tram.wagon_pos[1] = np.array([8.0, config.WORLD_HD - 0.10])
cw0, wh0 = app.world.cargo_wall_hits, app.world.wall_hits
app.tram._check_walls()
print(f"2б) Глубокий контакт: груз +{app.world.cargo_wall_hits - cw0}, "
      f"кузов +{app.world.wall_hits - wh0}")
ok &= app.world.cargo_wall_hits - cw0 == 1 and app.world.wall_hits - wh0 == 1

# 3) классификация: работник у кузова (−25) vs в зоне выступа груза (−20)
def isolate(ped):
    """Убираем остальных работников, чтобы тест был детерминированным."""
    for p in app.world.pedestrians:
        if p is not ped:
            p.pos = np.array([500.0, 500.0])

# 3а) работник вплотную к тягачу → НАЕЗД кузовом
place(0.0, 0.0, 0.0, cargo=True)
f = app.tram.fwd()
side = np.array([f[1], -f[0]])          # вправо от курса
ped = app.world.pedestrians[1]
isolate(ped)
app.tram.speed = 1.5
ped.pos = app.tram.pos + side * 0.7     # зона кузова паровозика (hw 0.5)
ped.knocked = 0.0
h0, cp0 = app.world.hits, app.world.cargo_ped_hits
app.tram._check_pedestrians()
print(f"3а) Кузов→работник: наездов +{app.world.hits - h0}, "
      f"груз-ударов +{app.world.cargo_ped_hits - cp0}")
ok &= app.world.hits == h0 + 1 and app.world.cargo_ped_hits == cp0

# 3б) работник в полосе выступа груза прицепа (вне зоны кузова) → ГРУЗ
place(0.0, 0.0, 0.0, cargo=True)
wp = app.tram.wagon_pos[1]
ped = app.world.pedestrians[2]
isolate(ped)
app.tram.speed = 1.5
ped.pos = wp + side * 0.83             # кузов 0.35+0.45=0.80 < 0.83 < 0.41+0.45=0.86
ped.knocked = 0.0
h0, cp0 = app.world.hits, app.world.cargo_ped_hits
app.tram._check_pedestrians()
print(f"3б) Груз→работник: наездов +{app.world.hits - h0}, "
      f"груз-ударов +{app.world.cargo_ped_hits - cp0}")
ok &= app.world.cargo_ped_hits == cp0 + 1 and app.world.hits == h0

# 4) кулдаун: одно и то же препятствие не штрафуется чаще WALL_COOLDOWN
r1 = app.world.register_wall_hit(False, "wall", "wall0")
r2 = app.world.register_wall_hit(False, "wall", "wall0")      # сразу — кулдаун
app.world.time += config.WALL_COOLDOWN + 0.1
r3 = app.world.register_wall_hit(False, "wall", "wall0")      # после кулдауна
r4 = app.world.register_wall_hit(True, "wall", "wall0")       # груз — свой кулдаун
print(f"4) Кулдаун: первый {r1}, повторный {r2}, после {config.WALL_COOLDOWN} с "
      f"{r3}, груз отдельный {r4}")
ok &= r1 and not r2 and r3 and r4

# 5) регресс: наезд кузовом — ровно −25, событие в очереди мира
ped = app.world.pedestrians[3]
ped.knocked = 0.0
s0, h0 = app.world.score, app.world.hits
app.world.events.clear()                 # убираем события предыдущих тестов
app.world.register_hit(ped, app.tram.pos)
ev = app.world.pop_events()
print(f"5) Регресс наезда: счёт {s0} → {app.world.score}, "
      f"событие: {ev[0][0] if ev else 'нет'}")
ok &= app.world.score == s0 - config.HIT_PENALTY
ok &= bool(ev) and "НАЕЗД" in ev[0][0]

print("ИТОГ:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
