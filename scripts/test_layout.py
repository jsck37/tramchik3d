#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тест адаптивной раскладки окон.

Проверяет:
  1) порядок колонок: слева лидары+камеры, справа карта→задание→статистика,
     YOLO-сетка в центре (только m1 активно по умолчанию);
  2) все видимые окна в границах экрана и не перекрываются;
  3) при включении ВСЕХ окон сетка 2×4 умещается в центр без перекрытий;
  4) ресайз пересчитывает раскладку; перетащенное окно (moved) сохраняется,
     F6 (reset_layout) возвращает всё на места;
  5) рендер на 5 разрешениях без ошибок + 2 скриншота 1600×900.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

import config  # noqa: E402
import ui.panels as panels  # noqa: E402
from main import App  # noqa: E402

import argparse  # noqa: E402

CHECKS = []


def ck(name, cond, extra=""):
    CHECKS.append((name, bool(cond)))
    print(("  [OK] " if cond else "  [FAIL] ") + name + (f"  {extra}" if extra else ""))


def rects_of(app, ids):
    return {wid: pygame.Rect(app.wm.index[wid].rect) for wid in ids}


def check_layout(app, W, H, tag):
    wm = app.wm
    left = ["lidar_top", "lidar_bot", "cam", "cam2"]
    right = ["minimap", "mission", "stats"]
    grid = panels.GRID_IDS
    R = rects_of(app, left + right + grid + ["help"])

    # --- 1. порядок левой колонки (сверху вниз) ---
    ys = [R[w].y for w in left]
    ck(f"{tag}: слева порядок лидарВ<лидарН<кам1<кам2", ys == sorted(ys), str(ys))
    ck(f"{tag}: слева x у левого края", all(R[w].x < W * 0.2 for w in left))
    # --- 2. порядок правой колонки ---
    ys = [R[w].y for w in right]
    ck(f"{tag}: справа карта<задание<статистика", ys == sorted(ys), str(ys))
    ck(f"{tag}: справа x у правого края", all(R[w].x > W * 0.75 for w in right))
    # задание строго НАД статистикой
    ck(f"{tag}: задание над статистикой",
       R["mission"].bottom <= R["stats"].y, f"{R['mission'].bottom} <= {R['stats'].y}")
    # --- 3. центр: m1 — в центральной зоне (между колонками), верхний ряд ---
    ck(f"{tag}: m1 в центре",
       R["m1"].x >= R["lidar_top"].right - 1 and
       R["m1"].right <= R["minimap"].x + 1 and R["m1"].y >= 30,
       f"x={R['m1'].x}, right={R['m1'].right}")
    # --- 4. границы экрана ---
    ok_b = all(r.x >= 0 and r.y >= 30 and r.right <= W and r.bottom <= H - 25
               for r in R.values())
    ck(f"{tag}: все окна в границах экрана", ok_b)
    # --- 5. видимые по умолчанию не перекрываются ---
    vis = [w.id for w in wm.wins if w.visible]
    bad = [(a, b) for i, a in enumerate(vis) for b in vis[i + 1:]
           if R[a].colliderect(R[b])]
    ck(f"{tag}: видимые окна не перекрываются ({len(vis)} шт.)", not bad, str(bad))
    # --- 6. по умолчанию только m1 из YOLO ---
    yolo_ids = {f"m{i}" for i in range(1, 8)}
    yolo_def = [w.id for w in wm.wins if w.visible and w.id in yolo_ids]
    ck(f"{tag}: YOLO по умолчанию только №1", yolo_def == ["m1"], str(yolo_def))
    # --- 7. все окна центральной сетки помещаются в центр ---
    x0, x1 = R["m1"].x, max(R[w].right for w in grid)
    ok_g = all(R[w].x >= R["lidar_top"].right and R[w].right <= R["minimap"].x
               for w in grid)
    ck(f"{tag}: сетка между колонками", ok_g,
       f"[{R['lidar_top'].right}..{R['minimap'].x}] vs [{x0}..{x1}]")
    return R


def main():
    print("ТЕСТ АДАПТИВНОЙ РАСКЛАДКИ (headless)")
    args = argparse.Namespace(selftest=False, cam=None, width=None, height=None)
    app = App(args)
    app.keys = {"up": False, "down": False, "left": False, "right": False,
                "brake": False}

    # --- рендер и проверки на 5 разрешениях ---
    sizes = [(1600, 900), (1280, 720), (1920, 1080), (1152, 648), (1024, 600)]
    for (W, H) in sizes:
        app._resize(W, H)
        for _ in range(12):
            app.update(1 / 60.0)
            app.render()
        check_layout(app, W, H, f"{W}x{H}")

    # --- все окна включены: сетка умещается (после переключения
    # окон раскладка пересчитывается — как по клавишам через app._toggle) ---
    app._resize(1600, 900)
    for wv in app.wm.wins:
        wv.visible = True
    panels.layout_all(app, force=True)
    for _ in range(12):
        app.update(1 / 60.0)
        app.render()
    R = rects_of(app, panels.GRID_IDS)
    bad = [(a, b) for i, a in enumerate(panels.GRID_IDS)
           for b in panels.GRID_IDS[i + 1:] if R[a].colliderect(R[b])]
    ck("все 8 окон сетки не перекрываются", not bad, str(bad))
    vis = [w for w in app.wm.wins if w.visible and w.id != "help"]  # help — оверлей
    Rall = {w.id: pygame.Rect(w.rect) for w in vis}
    bad2 = [(a, b) for i, a in enumerate(vis) for b in vis[i + 1:]
            if Rall[a.id].colliderect(Rall[b.id])]
    ck("ВСЕ 13 окон не перекрываются", not bad2, str(bad2))

    # --- moved-флаг и F6 ---
    app.wm.index["m2"].rect[0] += 333          # «перетащили» вручную
    app.wm.index["m2"].moved = True
    old_m2 = list(app.wm.index["m2"].rect)
    app._resize(1280, 720)
    ck("moved-окно переживает ресайз",
       list(app.wm.index["m2"].rect) == old_m2)
    ck("остальные окна пересчитались",
       app.wm.index["m1"].rect != app.wm.index["m2"].rect)
    panels.reset_layout(app)                    # F6
    grid_now = rects_of(app, ["m1", "m2"])
    ck("F6 возвращает moved-окно в сетку",
       grid_now["m2"].y == grid_now["m1"].y or
       grid_now["m2"].x != old_m2[0])
    ck("moved-флаги сброшены",
       not any(w.moved for w in app.wm.wins))

    # --- скриншоты (дефолт и все окна) 1600×900 ---
    app._resize(1600, 900)
    for wv in app.wm.wins:
        wv.visible = wv.id not in ("m2", "m3", "m4", "m5", "m6", "m7",
                                   "webcam", "help")
    app.wm.index["m1"].visible = True
    panels.layout_all(app, force=True)
    for _ in range(30):
        app.update(1 / 60.0)
        app.render()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p1 = os.path.join(root, "screenshot.png")
    pygame.image.save(app.screen, p1)
    ck("скриншот по умолчанию сохранён", os.path.exists(p1), p1)
    for wv in app.wm.wins:
        wv.visible = True
    app.wm.index["help"].visible = False
    panels.layout_all(app, force=True)
    for _ in range(30):
        app.update(1 / 60.0)
        app.render()
    p2 = os.path.join(root, "screenshot_all.png")
    pygame.image.save(app.screen, p2)
    ck("скриншот всех окон сохранён", os.path.exists(p2), p2)

    # --- FPS (информативно) ---
    t0 = time.perf_counter()
    for _ in range(120):
        app.update(1 / 60.0)
        app.render()
    ms = (time.perf_counter() - t0) / 120 * 1000.0
    print(f"  FPS (все окна, headless): {1000.0 / ms:.0f}")

    pygame.quit()
    fails = [n for n, okk in CHECKS if not okk]
    print(f"\nИТОГО: {len(CHECKS) - len(fails)}/{len(CHECKS)} проверок пройдено")
    if fails:
        print("ПРОВАЛЕНО:", ", ".join(fails))
        return 1
    print("РЕЗУЛЬТАТ: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
