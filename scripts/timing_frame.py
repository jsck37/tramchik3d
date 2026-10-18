# -*- coding: utf-8 -*-
"""Точный хронометраж частей кадра (без накладных cProfile)."""
import os, sys, time
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pygame
import config
import main as m

app = m.App(type("A", (), {"selftest": True, "cam": 0, "width": None,
                           "height": None})())
app.keys = {"up": False, "down": False, "left": False, "right": False,
            "brake": False}
for _ in range(60):
    app.update(1 / 60.0)
    app.render()

N = 240
t_world = t_npc_tram = t_lidars = t_cams = t_rest_u = 0.0
t_scene = t_3d_rest = t_win = t_bars = t_rest_r = 0.0
t_all = 0.0

for i in range(N):
    t0 = time.perf_counter()
    # --- разложение update ---
    a = time.perf_counter()
    app.world.update(1 / 60.0)
    b = time.perf_counter(); t_world += b - a
    app.npc.update(1 / 60.0)
    c = time.perf_counter(); t_npc_tram += c - b
    app.tram.update(1 / 60.0, app.keys)
    d = time.perf_counter(); t_npc_tram += d - c
    for lid in (app.lidar_top, app.lidar_bot):
        if app.frame_n % lid.every == 0 or not lid.scan:
            lid.update(app.world, app.tram)
    e = time.perf_counter(); t_lidars += e - d
    needed_front = app.wm.any_visible(
        ["cam"] + [f"m{i}" for i in range(1, 8)]) or app.recorder.active
    if needed_front and app.frame_n % app.onboard.every == 0:
        app.onboard.update(app)
    if app.wm.any_visible(["cam2"]) and app.frame_n % app.rear.every == 0:
        app.rear.update(app)
    f = time.perf_counter(); t_cams += f - e
    for t in app.toasts:
        t[1] -= 1 / 60.0
    app.toasts = [t for t in app.toasts if t[1] > 0]
    g = time.perf_counter(); t_rest_u += g - f
    # --- разложение render ---
    h = time.perf_counter()
    scene = app.scene
    direct = scene.get_size() == app.screen.get_size()
    target = app.screen if direct else scene
    if direct:
        app.screen.blit(app.sky, (0, 0))
    cd = app.cam.compute(*target.get_size())
    app.renderer.begin_view(target, cd)
    i0 = time.perf_counter()
    app.world.draw_scene(app.renderer, grid=True)
    i1 = time.perf_counter(); t_scene += i1 - i0
    app.npc.draw(app.renderer)
    app.tram.draw(app.renderer)
    app.tram.draw_sensors(app.renderer, app.lidar_top, app.lidar_bot)
    app.renderer.end_view()
    i2 = time.perf_counter(); t_3d_rest += i2 - i1
    app.wm.draw(app.screen)
    i3 = time.perf_counter(); t_win += i3 - i2
    app._bars(); app._toasts()
    i4 = time.perf_counter(); t_bars += i4 - i3
    t_all += time.perf_counter() - t0

ms = 1000.0
print(f"КАДР: {t_all / N * ms:.2f} мс ({1 / (t_all / N):.0f} FPS)")
print(f"update: world {t_world / N * ms:.2f} · npc+tram {t_npc_tram / N * ms:.2f}"
      f" · лидары {t_lidars / N * ms:.2f} · камеры {t_cams / N * ms:.2f}"
      f" · прочее {t_rest_u / N * ms:.2f}")
print(f"render: draw_scene {t_scene / N * ms:.2f} · 3D-прочее {t_3d_rest / N * ms:.2f}"
      f" · окна UI {t_win / N * ms:.2f} · панели {t_bars / N * ms:.2f}"
      f" · прочее {t_rest_r / N * ms:.2f}")
