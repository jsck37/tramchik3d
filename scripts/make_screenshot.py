# -*- coding: utf-8 -*-
"""Скриншоты: склад с реальными окнами → screenshot.png / screenshot_all.png

screenshot.png      — производственный вид: мини-карта, задание, статистика,
                      лидары, камеры, YOLO №1 (окна по умолчанию)
screenshot_all.png  — все 16 окон (стресс-вид)
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as m  # noqa: E402
import config  # noqa: E402
import pygame  # noqa: E402

# быстрые операции — чтобы кадры попали на движение/разгрузку
config.STOP_LINE_TIME = 1.0
config.LOAD_TIME = 4.0
config.UNLOAD_TIME = 5.0


class A:
    selftest = True
    cam = None
    width = None
    height = None


app = m.App(A())
app.keys = {"up": False, "down": False, "left": False, "right": False,
            "brake": False}
app.tram.timer = 1.0
app.tram.auto = True
# вид: изнутри склада A на площадку погрузки и малый круг
app.cam.yaw = -2.00         # камера в западной части склада
app.cam.pitch = 0.45
app.cam.dist = 12.0
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# --- производственный вид (окна по умолчанию) ---
for i in range(300):
    app.update(1 / 60)
    app.render()
pygame.image.save(app.screen, os.path.join(ROOT, "screenshot.png"))
print("сохранён screenshot.png")

# --- все окна ---
for wv in app.wm.wins:
    wv.visible = True
for i in range(120):
    app.update(1 / 60)
    app.render()
pygame.image.save(app.screen, os.path.join(ROOT, "screenshot_all.png"))
print("сохранён screenshot_all.png")
pygame.quit()
