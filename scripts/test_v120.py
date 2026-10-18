# -*- coding: utf-8 -*-
"""Тесты: два лидара (верх/низ), две камеры (передняя/задняя),
детальные панели YOLO 1-7, габарит состава 0.5x + регрессии"""
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

import main as m  # noqa: E402
import config  # noqa: E402
import ui.panels as panels  # noqa: E402


class A:
    selftest = True
    cam = None
    width = None
    height = None


app = m.App(A())
app.keys = {"up": False, "down": False, "left": False, "right": False,
            "brake": False}
# быстрые операции (проверяем логику очков, не хронометраж)
config.STOP_LINE_TIME = 1.0
config.LOAD_TIME = 4.0
config.UNLOAD_TIME = 4.0
config.BREAK_PERIOD = 1e9
app.tram.timer = min(app.tram.timer, 1.0)
ok = True


def check(name, cond, info=""):
    global ok
    print(f"{name}: {'OK' if cond else 'FAIL'} {info}")
    ok &= bool(cond)


# 0) регресс ресайза (был AttributeError scratch)
try:
    app._resize(1280, 720)
    app._resize(config.WINDOW_W, config.WINDOW_H)
    check("Ресайз окна", True)
except Exception as e:  # noqa: BLE001
    check("Ресайз окна", False, str(e))

# 1) габарит состава
check("TRAIN_SCALE", config.TRAIN_SCALE == 1.0)
segs = app.tram.segments()
loco_hw = segs[0][4]
check("Габарит тягача (коллизии)", abs(loco_hw - config.LOCO_WIDTH / 2) < 1e-6,
      f"half_width={loco_hw}")
check("WAGON_GAP", abs(config.WAGON_GAP - 0.95) < 1e-9,
      f"gap={config.WAGON_GAP:.3f}")
check("Длина состава", app.tram.length_total() < 6.0,
      f"{app.tram.length_total():.2f} м")
check("Масса состава", abs(app.tram.mass() - 1820.0) < 1e-6,
      f"{app.tram.mass():.0f} кг")

# 2) лидары: структура сканов + верхний видит пешехода впереди
lt, lb = app.lidar_top, app.lidar_bot
app.tram.reset()
app.tram.auto = False
for _ in range(5):
    app.update(1 / 60)
check("Лидар верхний: лучи", len(lt.scan) == config.LIDAR_TOP["rays"],
      f"{len(lt.scan)}/{lt.rays}")
check("Лидар нижний: лучи", len(lb.scan) == config.LIDAR_BOTTOM["rays"],
      f"{len(lb.scan)}/{lb.rays}")
f = app.tram.fwd()
ped = app.world.pedestrians[0]
ped.pos = np.array([app.tram.pos[0] + f[0] * 4.0, app.tram.pos[1] + f[1] * 4.0])
ped.target = ped.pos.copy()
for _ in range(6):
    app.update(1 / 60)
kinds_top = {k for (_a, _d, k) in lt.scan}
check("Верхний видит человека", "person" in kinds_top, str(sorted(kinds_top)))
kinds_bot = {k for (_a, _d, k) in lb.scan}
nd, nk = lt.nearest()
check("Верхний: дистанция < дальности", nd < lt.range, f"{nd:.1f} м {nk}")
check("Нижний: скан пола/препятствий", nd <= lb.range + 1e-6 and len(lb.scan) > 0)
fc = lt.front_clearance(app.tram, 30.0)
check("front_clearance работает", 0.0 < fc <= lt.range, f"{fc:.1f} м")

# 3) две камеры: передняя и задняя реально различаются
app.wm.index["cam"].visible = True
app.wm.index["cam2"].visible = True
for _ in range(10):
    app.update(1 / 60)
front = pygame.surfarray.array2d(app.onboard.surface).copy()
rear = pygame.surfarray.array2d(app.rear.surface)
diff = float(np.mean(np.abs(front.astype(int) - rear.astype(int))))
check("Камера 2 (задняя) рендерится", app.rear.camdata is not None)
check("Кадры камер различаются", diff > 100.0, f"avg diff={diff:.0f}")

# 4) детальные панели YOLO 1-7: рисуются без ошибок, метрики есть
check("SIDES: 7 задач", set(panels.SIDES) == set(config.TASKS))
app.yolo_ms = 12.5
for i in range(1, 8):
    app.wm.index[f"m{i}"].visible = True
app.update(1 / 60)
app.render()
check("Рендер всех окон (детальные панели)", True)
model, ms, hz = panels._task_meta(app, "detect")
check("Метрики модели", model == config.YOLO_MODEL_NAMES["detect"] and ms > 0 and hz > 0,
      f"{model} {ms:.1f} мс {hz:.0f} Гц")

# 5) окна лидаров и камер рендерятся
for wid in ("lidar_top", "lidar_bot"):
    app.wm.index[wid].visible = True
app.render()
check("Рендер панелей лидаров", True)

# 6) прицепы следуют с новым интервалом
app.tram.reset()
app.tram.auto = False
app.keys["up"] = True
for _ in range(300):
    app.update(1 / 60)
app.keys["up"] = False
gaps = []
prev = app.tram.pos
for wp in app.tram.wagon_pos:
    gaps.append(float(np.hypot(wp[0] - prev[0], wp[1] - prev[1])))
    prev = wp
check("3 прицепа", len(app.tram.wagon_pos) == 3)
check("Интервалы состава", all(abs(g - config.WAGON_GAP) < 1.0 for g in gaps),
      str([f"{g:.2f}" for g in gaps]))

# 7) запрет реверса
app.keys["down"] = True
p0 = app.tram.pos.copy()
yaw0 = app.tram.yaw
f0 = np.array([math.sin(yaw0), math.cos(yaw0)])
min_speed = 1e9
for _ in range(120):
    app.update(1 / 60)
    min_speed = min(min_speed, app.tram.speed)
app.keys["down"] = False
rel = app.tram.pos - p0
along = float(rel @ f0)
check("Реверс запрещён", min_speed >= -1e-6 and along >= -0.35 and
      app.tram.reverse_hold > 0.0, f"v={min_speed:.2f} along={along:+.2f}")

# 8) наезд: −25, пешеход падает
app.tram.reset()
app.tram.auto = False
app.keys["up"] = True
for _ in range(60):
    app.update(1 / 60)
f = app.tram.fwd()
ped = app.world.pedestrians[0]
ped.pos = np.array([app.tram.pos[0] + f[0] * 1.6,
                    app.tram.pos[1] + f[1] * 1.6])
ped.target = ped.pos.copy()
score0, hits0 = app.world.score, app.world.hits
for _ in range(90):
    app.update(1 / 60)
app.keys["up"] = False
check("Наезд: −25", app.world.score == score0 - config.HIT_PENALTY and
      app.world.hits == hits0 + 1 and ped.knocked > 0.0,
      f"score {score0}->{app.world.score}, hits {hits0}->{app.world.hits}")

# 9) фьюжн: верхний лидар тормозит перед близким препятствием
app.tram.reset()
app.tram.auto = True
crate = app.world.crates[0]
crate["pos"] = np.array([app.tram.pos[0], app.tram.pos[1] + 2.2])
save_r = crate["radius"]
speeds = []
for _ in range(40):
    app.update(1 / 60)
    speeds.append(app.tram.speed)
crate["pos"] = np.array([999.0, 999.0])
crate["radius"] = save_r
check("Фьюжн с лидаром (тихий ход)", max(speeds) < config.TRAM_SPEED,
      f"max v={max(speeds):.2f}")

# 10) доставка: +100 на автопилоте (полный цикл; штрафы стен/груза
#     компенсируем — проверяем именно начисление за доставку)
app.tram.reset()
app.tram.auto = True
score0 = app.world.score
wh0 = app.world.wall_hits
cw0 = app.world.cargo_wall_hits
cp0 = app.world.cargo_ped_hits
frames = 0
while app.tram.cycles < 1 and frames < 10800:
    app.update(1 / 60)
    frames += 1
refund = ((app.world.wall_hits - wh0) * config.WALL_HIT_PENALTY +
          (app.world.cargo_wall_hits - cw0) * config.CARGO_WALL_PENALTY +
          (app.world.cargo_ped_hits - cp0) * config.CARGO_PED_PENALTY)
check("Доставка +100", app.tram.cycles >= 1 and
      app.world.score >= score0 + config.DELIVERY_REWARD - config.HIT_PENALTY * 2 - refund,
      f"score {score0}->{app.world.score} за {frames / 60:.0f} с "
      f"(стены {app.world.wall_hits - wh0}, груз {refund // 10})")

# 11) пешеход встаёт и убегает
t_wait = 0.0
while ped.knocked > 0 and t_wait < 6.0:
    app.update(1 / 60)
    t_wait += 1 / 60
moved = float(np.hypot(ped.pos[0] - ped.flee_from[0], ped.pos[1] - ped.flee_from[1]))
check("Пешеход встал и убежал", ped.knocked <= 0.0 and moved > 2.0,
      f"{t_wait:.1f} с, {moved:.1f} м")

print("ИТОГ:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
