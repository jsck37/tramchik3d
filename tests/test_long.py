# -*- coding: utf-8 -*-
"""Долгий прогон: A*, восприятие, полный цикл доставки, ручной режим, датасет."""
import math
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as m  # noqa: E402
import config  # noqa: E402


class A:
    selftest = True
    cam = None
    width = None
    height = None


app = m.App(A())
app.keys = {"up": False, "down": False, "left": False, "right": False, "brake": False}
# быстрые операции (функциональный тест, не хронометраж)
config.STOP_LINE_TIME = 1.0
config.LOAD_TIME = 4.0
config.UNLOAD_TIME = 4.0
config.BREAK_PERIOD = 1e9
app.tram.timer = min(app.tram.timer, 1.0)
ok = True

# 1) веб-камера без физической камеры не должна падать
from perception.webcam import WebcamSource, DatasetRecorder  # noqa: E402
src = WebcamSource(99)
print("WebcamSource без камеры:", "OK" if not src.ok and src.err else "FAIL")
ok &= not src.ok
rec = DatasetRecorder()
ok &= os.path.isdir(rec.img_dir)
print("DatasetRecorder: каталоги созданы:", os.path.isdir(rec.img_dir))

# 2) восприятие из стартовой позиции (курс на склад — внутрь, вид на работников)
app.tram.yaw = -math.pi / 2          # разворачиваем нос внутрь склада A
app.onboard.update(app)
out = app.onboard.outputs
nonempty = {k: len(v) for k, v in out.items()}
print("Восприятие (7 задач):", nonempty)
ok &= all(len(v) > 0 for v in out.values()) and len(out) == 7

from perception.sim_perception import to_yolo_lines  # noqa: E402
lines = to_yolo_lines(out, 480, 320)
print(f"YOLO-меток из кадра: {len(lines)}; пример: {lines[0] if lines else '-'}")
ok &= len(lines) > 0

# 3) сохранение кадра датасета из симуляции
p = rec.save_pygame_surface(app.onboard.surface, lines, tag="sim")
exists = p and os.path.exists(p) and os.path.getsize(p) > 0
print("Кадр датасета сохранён:", exists, "->", p)
ok &= bool(exists)
rec.ensure_data_yaml()
ok &= os.path.exists(os.path.join(config.DATASET_DIR, "data.yaml"))

# 4) ручной режим из чистой позиции (сброс в зону A, курс на центр)
app.tram.reset()
app.tram.auto = False
p0 = app.tram.pos.copy()
app.keys["up"] = True
for _ in range(180):
    app.update(1 / 60)
app.keys["up"] = False
d = float(math.hypot(app.tram.pos[0] - p0[0], app.tram.pos[1] - p0[1]))
print(f"Ручной режим: проехано {d:.1f} м за 3 с (ожидалось ~6 м при 2,2 м/с)")
ok &= d > 4.0
app.keys["left"] = True
yaw0 = app.tram.yaw
for _ in range(60):
    app.update(1 / 60)
app.keys["left"] = False
print(f"Поворот: dYaw = {app.tram.yaw - yaw0:+.2f} рад")
ok &= abs(app.tram.yaw - yaw0) > 0.5

# 5) автопилот: 2 полных рейса (Р-Т-Т-Т: A → 150 м → B → малый круг → A);
# ускорение времени ×8 — полный цикл ~170 с симуляции
app.tram.reset()
app.tram.auto = True
app.time_scale = 8
t0 = time.perf_counter()
frames = 0
while app.tram.cycles < 2 and frames < 36000:
    app.update(1 / 60)
    app.render()
    frames += 1
app.time_scale = 1
print(f"Рейсов: {app.tram.cycles} за {frames / 60:.0f} с кадров "
      f"({frames * 8 / 60:.0f} с сим., реальное время {time.perf_counter() - t0:.1f} с), "
      f"состояние: {app.tram.state}")
print(f"робот II: скорость {app.npc.speed:.1f} м/с · штрафов с ним: "
      f"{app.world.npc_hits}")
ok &= app.tram.cycles >= 2

# 6) A* на нескольких парах точек (внутри складов и через территорию)
pairs = [((config.ZONE_A), (config.ZONE_B)),
         ((-9.0, 4.3), (12.0, -4.5)),
         ((0.0, -4.5), (0.0, 4.5))]
for (a, b) in pairs:
    path = app.world.find_path(a, b)
    print(f"A* {a} -> {b}: {len(path)} точек")
    ok &= len(path) >= 2

print("ИТОГ:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
