# -*- coding: utf-8 -*-
"""Диагностика медленной доставки: скорость, стопы, колпачки по фазам."""
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
app.tram.reset()
app.tram.auto = True

# фазы: TO_B / TO_A — едем; LOADING/UNLOADING — стоим
phase_time = {}
stop_frames = 0
cap_frames = 0
frames = 0
speed_sum = 0.0
stuck = []
last_pos = app.tram.pos.copy()
last_move_t = 0.0

while frames < 10800 and app.tram.cycles < 2:
    prev_state = app.tram.state
    app.update(1 / 60)
    phase_time[prev_state] = phase_time.get(prev_state, 0.0) + 1 / 60
    speed_sum += app.tram.speed
    # останавливался ли из-за пешехода
    if app.tram.auto and app.tram.speed < 0.05 and \
       prev_state in ("ДВИЖЕНИЕ К B", "ВОЗВРАЩЕНИЕ К A"):
        # проверить, есть ли пешеход впереди
        f = app.tram.fwd()
        for ped in app.world.pedestrians:
            diff = ped.pos - app.tram.pos
            d = math.hypot(diff[0], diff[1])
            if d < 5.0 and (diff[0] * f[0] + diff[1] * f[1]) / max(d, 1e-4) > 0.72:
                stop_frames += 1
                break
    if abs(app.tram.speed - 2.64) < 0.35:
        cap_frames += 1
    if np.hypot(*(app.tram.pos - last_pos)) > 0.3:
        last_pos = app.tram.pos.copy()
        last_move_t = frames / 60.0
    elif frames / 60.0 - last_move_t > 8.0:
        f = app.tram.fwd()
        blockers = []
        for ped in app.world.pedestrians:
            diff = ped.pos - app.tram.pos
            d = math.hypot(diff[0], diff[1])
            if d < 5.0 and (diff[0] * f[0] + diff[1] * f[1]) / max(d, 1e-4) > 0.72:
                blockers.append((round(d, 1), ped.id, ped.knocked,
                                 [round(float(x), 1) for x in ped.pos],
                                 [round(float(x), 1) for x in ped.target],
                                 round(ped.speed, 2)))
        stuck.append((round(frames / 60.0, 1), app.tram.state,
                      [round(float(x), 1) for x in app.tram.pos], blockers))
        last_move_t = frames / 60.0
        last_pos = app.tram.pos.copy()
    frames += 1

print("фазы:", {k: f"{v:.0f}с" for k, v in phase_time.items()})
print(f"циклов: {app.tram.cycles} за {frames / 60:.0f} с, средняя скорость "
      f"{speed_sum / max(1, frames):.2f} м/с")
print(f"кадров в стопе-из-за-пешехода: {stop_frames} ({stop_frames / 60:.0f} с), "
      f"кадров на колпачке 0.6: {cap_frames} ({cap_frames / 60:.0f} с)")
print("зависания >8с:", stuck[:10])
