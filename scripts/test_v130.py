# -*- coding: utf-8 -*-
"""Тесты: паровозик с 3 прицепами, запрет реверса, штрафы за наезд,
очки за доставку, ресайз окна (регресс ensure_scratch)."""
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
# быстрые операции (проверяем логику очков, не хронометраж)
config.STOP_LINE_TIME = 1.0
config.LOAD_TIME = 4.0
config.UNLOAD_TIME = 4.0
config.BREAK_PERIOD = 1e9
app.tram.timer = min(app.tram.timer, 1.0)
ok = True

# 0) регресс: _resize не должен падать (был AttributeError: 'WindowManager'
#    object has no attribute 'scratch')
try:
    app._resize(1280, 720)
    app._resize(config.WINDOW_W, config.WINDOW_H)
    print("Ресайз окна: OK")
except Exception as e:  # noqa: BLE001
    print("Ресайз окна: FAIL —", e)
    ok = False

# включим все окна и отрендерим кадр — проверка панелей/миникарты
for wv in app.wm.wins:
    wv.visible = True
app.update(1 / 60)
app.render()
print("Рендер со всеми окнами: OK")

# 1) состав: 3 прицепа следуют по следу с интервалом WAGON_GAP
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
print(f"Прицепов: {len(app.tram.wagon_pos)}, интервалы: "
      f"{['%.2f' % g for g in gaps]} (цель {config.WAGON_GAP}, "
      f"тягач→вагон {config.GAP_LOCO_WAGON})")
ok &= len(app.tram.wagon_pos) == 3
ok &= all(abs(g - config.WAGON_GAP) < 1.3 for g in gaps[1:])
ok &= abs(gaps[0] - config.GAP_LOCO_WAGON) < 0.3

# 2) запрет реверса: S — только тормоз; назад состав не едет
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
print(f"Реверс: скорость {app.tram.speed:.2f} (мин {min_speed:.2f}), "
      f"смещение вдоль курса {along:+.2f} м, индикатор {app.tram.reverse_hold:.1f}")
ok &= min_speed >= -1e-6 and along >= -0.35 and app.tram.reverse_hold > 0.0

# 3) наезд на пешехода: −25 очков, пешеход падает; повторно не штрафуется
#    (состав 0.5x — пешеход ставим ближе, на границе габарита)
app.tram.reset()
app.tram.auto = False
app.keys["up"] = True
for _ in range(60):
    app.update(1 / 60)
f = app.tram.fwd()
ped = app.world.pedestrians[0]
ped.pos = np.array([app.tram.pos[0] + f[0] * 1.1,
                    app.tram.pos[1] + f[1] * 1.1])
ped.target = ped.pos.copy()
score0 = app.world.score
hits0 = app.world.hits
for _ in range(90):
    app.update(1 / 60)
app.keys["up"] = False
hit_event = any("НАЕЗД" in t[0] for t in app.toasts)
print(f"Наезд: счёт {score0} → {app.world.score}, наездов {hits0} → "
      f"{app.world.hits}, лежит ещё {ped.knocked:.1f} с, тост: {hit_event}")
ok &= app.world.score == score0 - config.HIT_PENALTY
ok &= app.world.hits == hits0 + 1 and ped.knocked > 0.0
# повторный проезд по лежащему — штраф только один
app.keys["up"] = True
for _ in range(60):
    app.update(1 / 60)
app.keys["up"] = False
ok &= app.world.hits == hits0 + 1

# 4) доставка: +100 очков на автопилоте (штрафы стен/груза)
app.tram.reset()
app.tram.auto = True
hits0 = app.world.hits
score0 = app.world.score
wh0 = app.world.wall_hits
cw0 = app.world.cargo_wall_hits
cp0 = app.world.cargo_ped_hits
frames = 0
delivered = False
while frames < 10800:
    app.update(1 / 60)
    app.render()          # заодно гоняем полный путь рендера
    frames += 1
    if app.tram.cycles >= 1:
        delivered = True
        break
wd = app.world
refund = ((wd.wall_hits - wh0) * config.WALL_HIT_PENALTY +
          (wd.cargo_wall_hits - cw0) * config.CARGO_WALL_PENALTY +
          (wd.cargo_ped_hits - cp0) * config.CARGO_PED_PENALTY)
gain = (app.world.score - score0
        + (app.world.hits - hits0) * config.HIT_PENALTY + refund)
print(f"Доставка: циклов {app.tram.cycles} за {frames / 60:.0f} с сим, "
      f"счёт {app.world.score:+d} (было {score0:+d}), начислено {gain:+d} "
      f"[стен {wd.wall_hits - wh0}, груз-событий {refund // 10}]")
ok &= delivered
ok &= gain >= config.DELIVERY_REWARD

# 5) пешеход встаёт и убегает
t_wait = 0.0
while ped.knocked > 0 and t_wait < 6.0:
    app.update(1 / 60)
    t_wait += 1 / 60
moved_away = 0.0
if ped.flee_from is not None:
    moved_away = float(np.hypot(ped.pos[0] - ped.flee_from[0],
                                ped.pos[1] - ped.flee_from[1]))
print(f"Пешеход поднялся за {t_wait:.1f} с, убежал на {moved_away:.1f} м от места наезда")
ok &= ped.knocked <= 0.0 and moved_away > 2.0

print("ИТОГ:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
