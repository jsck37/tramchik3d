#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""«Трамвайчик 3D» — ДВА склада и маршрут 150 м на собственном программном 3D-движке.

Запуск:        python main.py
Самопроверка:  python main.py --selftest   (headless, замер FPS)
"""
import argparse
import math
import os
import sys
import time
from collections import deque

import numpy as np
import pygame

import config
import ui.panels as panels
from engine.camera import Camera3D
from engine.renderer import Renderer
from perception.sim_perception import compute_outputs, to_yolo_lines
from perception.webcam import WebcamSource, RealYOLO, DatasetRecorder
from sim.lidar import Lidar
from sim.npc import NPCTram
from sim.tram import Tram
from sim.world import World
from ui.windows import WindowManager

TEXT = config.PAL["text"]
DIM = config.PAL["text_dim"]
ACC = config.PAL["accent"]
WARN = config.PAL["warn"]
OK = config.PAL["ok"]


class SensorCam:
    """Виртуальная камера на составе: CAM-1 передняя и CAM-2 задняя.

    Вьюпорт рендерится раз в every кадров; масштабированные копии для панелей
    кэшируются между кадрами. Для CAM-1 (perception=True) здесь же считается
    выход 7 задач YOLO (ground truth), время прохода и треки детекций."""

    def __init__(self, name, pose_fn, w, h, fov, every=1, perception=False):
        self.name = name
        self.pose_fn = pose_fn
        self.w, self.h = w, h
        self.fov = fov
        self.every = max(1, every)
        self.perception = perception
        self.cam3d = Camera3D(fov=fov, mode="fixed")
        self.surface = pygame.Surface((w, h))
        self.outputs = None
        self.camdata = None
        self.ms = 0.0            # скользящее время прохода восприятия, мс
        self.trails = {}         # id пешехода → deque центров bbox (панель detect)
        self._scaled = {}
        self._n = 0

    def scaled(self, size):
        img = self._scaled.get(size)
        if img is None:
            img = pygame.transform.scale(self.surface, size)
            self._scaled[size] = img
            if len(self._scaled) > 10:
                self._scaled.clear()
                img = pygame.transform.scale(self.surface, size)
                self._scaled[size] = img
        return img

    def update(self, app):
        surf = self.surface
        self._scaled = {}
        surf.fill((24, 28, 40))
        pos, fwd = self.pose_fn(app.tram)
        self.cam3d.set_fixed(pos, fwd)
        cd = self.cam3d.compute(self.w, self.h)
        self.camdata = cd
        r = app.renderer
        r.begin_view(surf, cd)
        app.world.draw_scene(r, grid=False)
        app.tram.draw(r)
        r.end_view()

        self._n += 1
        if self.perception and (self.outputs is None or self._n % 2 == 0):
            t0 = time.perf_counter()

            def project(p):
                dx = p[0] - cd.pos[0]
                dy = p[1] - cd.pos[1]
                dz = p[2] - cd.pos[2]
                x = dx * cd.u[0] + dy * cd.u[1] + dz * cd.u[2]
                y = dx * cd.v[0] + dy * cd.v[1] + dz * cd.v[2]
                z = dx * cd.w[0] + dy * cd.w[1] + dz * cd.w[2]
                if z <= 0.08:
                    return None
                sx = cd.cx + cd.focal * x / z
                sy = cd.cy - cd.focal * y / z
                if -40 <= sx <= self.w + 40 and -40 <= sy <= self.h + 40:
                    return (sx, sy, z)
                return None

            self.outputs = compute_outputs(app.world, app.tram, project,
                                           self.w, self.h, app.world.time)
            self.ms = self.ms * 0.8 + (time.perf_counter() - t0) * 800.0 * 0.2
            # треки детекций для панели detect (хвосты движения)
            for det in self.outputs.get("detect", []):
                pid = det.get("id")
                if pid is None:
                    continue
                bx, by, bw, bh = det["box"]
                tr = self.trails.setdefault(pid, deque(maxlen=16))
                tr.append((bx + bw / 2, by + bh / 2))
            if len(self.trails) > 32:
                self.trails.clear()
        # прицел «камеры»
        cx, cy = self.w // 2, self.h // 2
        pygame.draw.circle(surf, (240, 240, 240), (cx, cy), 4, 1)
        pygame.draw.line(surf, (240, 240, 240), (cx - 9, cy), (cx - 3, cy))
        pygame.draw.line(surf, (240, 240, 240), (cx + 3, cy), (cx + 9, cy))


class App:
    def __init__(self, args):
        self.args = args
        if args.selftest:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        flags = pygame.RESIZABLE
        size = (args.width or config.WINDOW_W, args.height or config.WINDOW_H)
        self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption(
            f"Трамвайчик 3D v{config.APP_VERSION} — виртуальный склад (собственный движок)")
        self.w, self.h = size
        self._font_cache = {}
        self.clock = pygame.time.Clock()

        # --- мир и акторы ---
        self.world = World()
        self.tram = Tram(self.world)
        self.world.tram = self.tram   # пешеходам нужен состав (обход)
        self.npc = NPCTram(self.world)     # робот II — второй состав
        self.world.npc = self.npc
        self.cam = Camera3D(fov=62.0)
        self.cam_follow = True
        self.renderer = Renderer()

        # --- датчики (v1.2.0): две камеры + два лидара ---
        self.onboard = SensorCam("CAM-1 · передняя", lambda t: t.cam_pose(),
                                 config.ONBOARD_W, config.ONBOARD_H,
                                 config.ONBOARD_FOV, every=config.ONBOARD_EVERY,
                                 perception=True)
        self.rear = SensorCam("CAM-2 · задняя", lambda t: t.rear_pose(),
                              config.CAM2_W, config.CAM2_H, config.CAM2_FOV,
                              every=config.CAM2_EVERY)
        self.lidar_top = Lidar(config.LIDAR_TOP, "ВЕРХНИЙ · 360°")
        self.lidar_bot = Lidar(config.LIDAR_BOTTOM, "НИЖНИЙ · 120°")
        self.world.lidar_top = self.lidar_top      # фьюжн с автопилотом
        self.yolo_ms = 0.0                          # конвейер восприятия, мс

        # --- UI ---
        self.wm = WindowManager(self)
        panels.register_all(self)

        # --- ускорение времени и вид камеры ---
        self.time_scale = 1                # ×1/×2/×4/×8 (клавиша T)
        self.view_mode = "orbit"           # orbit | chase | cab (клавиша F)

        # --- веб-камера / датасет ---
        self.webcam = None
        self.real_yolo = RealYOLO()
        self.recorder = DatasetRecorder()
        self.cam_labels = []

        # --- служебное ---
        self.sky = self._make_sky(self.w, self.h)
        self.scene = pygame.Surface((int(self.w * config.RENDER_SCALE),
                                     int(self.h * config.RENDER_SCALE)))
        self.toasts = []
        self.t = 0.0
        self.frame_n = 0
        self.fps = 0.0
        self.fps_avg = 0.0
        self.fps_min = 9999.0
        self.frame_ms = 0.0
        self._fps_accum = []
        self.orbiting = False

    # --------------------------------------------------------------- шрифты
    _FONT_FILES = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self._font_cache:
            fnt = None
            for p in self._FONT_FILES:
                if os.path.exists(p):
                    try:
                        fnt = pygame.font.Font(p, size)
                        fnt.set_bold(bold)
                        break
                    except Exception:  # noqa: BLE001
                        continue
            if fnt is None:
                fnt = pygame.font.SysFont(
                    "dejavusansmono,consolas,menlo,dejavusans,arial", size, bold=bold)
            self._font_cache[key] = fnt
        return self._font_cache[key]

    def toast(self, text, color=(240, 244, 252)):
        self.toasts.append([text, 2.4, color])
        if len(self.toasts) > 5:
            self.toasts.pop(0)

    def _make_sky(self, w, h):
        surf = pygame.Surface((w, h))
        top, bot = config.PAL["sky_top"], config.PAL["sky_bot"]
        for i in range(0, h, 4):
            k = i / max(1, h)
            col = tuple(int(top[j] + (bot[j] - top[j]) * k) for j in range(3))
            pygame.draw.rect(surf, col, (0, i, w, 4))
        return surf

    # ----------------------------------------------------------------- цикл
    def run(self):
        if self.args.selftest:
            return self._selftest()
        running = True
        while running:
            dt_ms = self.clock.tick(config.FPS_CAP)
            dt = min(0.05, dt_ms / 1000.0) or 0.016
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    running = self._keydown(e)
                elif e.type == pygame.VIDEORESIZE:
                    self._resize(e.w, e.h)
                elif not self.wm.handle_event(e):
                    self._mouse(e)
            self._collect_keys()
            self.update(dt)
            self.render()
            pygame.display.flip()
        if self.webcam:
            self.webcam.close()
        pygame.quit()
        return 0

    # -------------------------------------------------------------- события
    def _toggle(self, wid):
        """Переключить окно и пересчитать адаптивную сетку YOLO (v1.6.0)."""
        self.wm.toggle(wid)
        panels.layout_all(self)

    def _cycle_time_scale(self):
        scales = config.TIME_SCALES
        i = scales.index(self.time_scale) if self.time_scale in scales else 0
        self.time_scale = scales[(i + 1) % len(scales)]
        self.toast(f"Ускорение времени: ×{self.time_scale}", (120, 220, 255))

    def _cycle_view(self):
        order = ["orbit", "chase", "cab"]
        self.view_mode = order[(order.index(self.view_mode) + 1) % len(order)]
        if self.view_mode == "orbit":
            self.cam.mode = "orbit"       # вернуть орбитальное управление
        names = {"orbit": "ОРБИТА (мышь)", "chase": "СЛЕДЯЩАЯ за составом",
                 "cab": "КАБИНА (1-е лицо, фиксированный)"}
        self.toast("Вид: " + names[self.view_mode], (120, 220, 255))

    def _keydown(self, e):
        k = e.key
        if k == pygame.K_ESCAPE:
            return False
        if k == pygame.K_F11:
            try:
                pygame.display.toggle_fullscreen()
                w, h = self.screen.get_size()
                self._resize(w, h)
            except Exception:  # noqa: BLE001
                pass
        elif k == pygame.K_F1:
            self._toggle("help")
        elif k == pygame.K_F2:
            self._toggle("stats")
        elif k == pygame.K_F3:
            self._toggle("minimap")
        elif k == pygame.K_F4:
            self._toggle("mission")
        elif k == pygame.K_F6:
            panels.reset_layout(self)
            self.toast("Раскладка: все окна упорядочены (адаптивно)")
        elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                   pygame.K_5, pygame.K_6, pygame.K_7):
            self._toggle(f"m{k - pygame.K_0}")
        elif k == pygame.K_8:
            self._toggle("lidar_top")
        elif k == pygame.K_9:
            self._toggle("lidar_bot")
        elif k == pygame.K_x:
            self._toggle("cam2")
        elif k == pygame.K_c:
            self._toggle("cam")
        elif k == pygame.K_v:
            self._toggle("webcam")
            if self.wm.index["webcam"].visible and self.webcam is None:
                self.toast("Подключаю веб-камеру...")
                self.webcam = WebcamSource(self.args.cam)
                self.toast("Камера включена" if self.webcam.ok
                           else f"Камера недоступна: {self.webcam.err}")
        elif k == pygame.K_t:
            self._cycle_time_scale()
        elif k == pygame.K_f:
            self._cycle_view()
        elif k == pygame.K_o:
            self.recorder.active = not self.recorder.active
            self.toast("Запись датасета: " +
                       ("ВКЛ (каждые %.1f с)" % config.REC_INTERVAL
                        if self.recorder.active else "ВЫКЛ"))
        elif k == pygame.K_TAB:
            self.tram.auto = not self.tram.auto
            self.toast("Режим: " + ("АВТОПИЛОТ (A→B, обход препятствий)"
                                    if self.tram.auto else "РУЧНОЙ (WASD/стрелки)"))
        elif k == pygame.K_m:
            if self.view_mode == "orbit":
                self.cam_follow = not self.cam_follow
                self.toast("Камера: " + ("следящая" if self.cam_follow
                                          else "свободная"))
            else:
                self._cycle_view()
        elif k == pygame.K_r:
            self.tram.reset()
            self.cam.target = np.array([self.tram.pos[0], 1.4, self.tram.pos[1]])
            self.toast("Сброс: трамвайчик на площадке A (склад 1)")
        return True

    def _mouse(self, e):
        if self.view_mode != "orbit":
            return                  # в следящем виде и кабине мышь не вращает
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            self.orbiting = True
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.orbiting = False
        elif e.type == pygame.MOUSEMOTION and self.orbiting:
            self.cam.orbit(e.rel[0], e.rel[1])
        elif e.type == pygame.MOUSEWHEEL:
            self.cam.zoom(e.y)

    def _collect_keys(self):
        kk = pygame.key.get_pressed()
        self.keys = {
            "up": kk[pygame.K_w] or kk[pygame.K_UP],
            "down": kk[pygame.K_s] or kk[pygame.K_DOWN],
            "left": kk[pygame.K_a] or kk[pygame.K_LEFT],
            "right": kk[pygame.K_d] or kk[pygame.K_RIGHT],
            "brake": kk[pygame.K_SPACE],
        }

    def _resize(self, w, h):
        self.w, self.h = w, h
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        self.sky = self._make_sky(w, h)
        self.scene = pygame.Surface((int(w * config.RENDER_SCALE),
                                     int(h * config.RENDER_SCALE)))
        self.wm.ensure_scratch(w, h)
        panels.layout_all(self)   # v1.4.0: адаптивная раскладка окон

    # -------------------------------------------------------------- обновление
    def update(self, dt):
        self.t += dt
        self.frame_n += 1
        # ускорение времени: N подшагов физики → процессы ×N,
        # рендер/сенсоры остаются реального времени
        for _ in range(self.time_scale):
            self.world.update(dt)
            self.npc.update(dt)               # робот II
            self.tram.update(dt, self.keys)
        # события мира (доставка/наезды) → цветные тосты
        for text, col in self.world.pop_events():
            self.toast(text, col)
        if self.cam_follow and self.view_mode == "orbit":
            self.cam.target = np.array([self.tram.pos[0], 1.2, self.tram.pos[1]])
        # лидары — каждый на своей частоте
        for lid in (self.lidar_top, self.lidar_bot):
            if self.frame_n % lid.every == 0 or not lid.scan:
                lid.update(self.world, self.tram)
        # виртуальные камеры — рендер только если видно их окна/идёт запись
        needed_front = self.wm.any_visible(
            ["cam"] + [f"m{i}" for i in range(1, 8)]) or self.recorder.active
        if needed_front and self.frame_n % self.onboard.every == 0:
            self.onboard.update(self)
        if self.wm.any_visible(["cam2"]) and self.frame_n % self.rear.every == 0:
            self.rear.update(self)
        if self.onboard.outputs is not None:
            self.yolo_ms = self.onboard.ms
        # запись датасета
        if self.recorder.active:
            self.recorder._timer += dt
            if self.recorder._timer >= config.REC_INTERVAL:
                self.recorder._timer = 0.0
                self._save_dataset_frame()
        for t in self.toasts:
            t[1] -= dt
        self.toasts = [t for t in self.toasts if t[1] > 0]

    def _save_dataset_frame(self):
        frame = self.webcam.read() if self.webcam and self.webcam.ok else None
        if frame is not None and self.real_yolo.available:
            lines = self.cam_labels
            p = self.recorder.save_cv2_frame(frame, lines)
        else:
            if self.onboard.outputs is None:
                self.onboard.update(self)
            lines = to_yolo_lines(self.onboard.outputs or {},
                                  config.ONBOARD_W, config.ONBOARD_H)
            p = self.recorder.save_pygame_surface(self.onboard.surface, lines, tag="sim")
        if p:
            self.recorder.ensure_data_yaml()

    # ---------------------------------------------------------------- рендер
    def render(self):
        t0 = time.perf_counter()
        scene = self.scene
        # при RENDER_SCALE=1 рисуем сразу на экран (без лишнего блита)
        direct = scene.get_size() == self.screen.get_size()
        target = self.screen if direct else scene
        if not direct:
            scene.blit(pygame.transform.scale(self.sky, scene.get_size()), (0, 0))
        else:
            self.screen.blit(self.sky, (0, 0))
        # --- виды камеры: орбита / следящая / кабина (1-е лицо) ---
        if self.view_mode == "cab":
            t = self.tram
            f = t.fwd()
            pos = np.array([t.pos[0] + f[0] * config.CAB_VIEW_FWD,
                            config.CAB_VIEW_HEIGHT,
                            t.pos[1] + f[1] * config.CAB_VIEW_FWD])
            self.cam.set_fixed(pos, (f[0], -0.055, f[1]), fov=74.0)
        elif self.view_mode == "chase":
            t = self.tram
            f = t.fwd()
            pos = np.array([t.pos[0] - f[0] * 7.5, 3.6, t.pos[1] - f[1] * 7.5])
            self.cam.set_fixed(pos, (f[0], -0.24, f[1]), fov=62.0)
        cd = self.cam.compute(*target.get_size())
        self.renderer.begin_view(target, cd)
        self.world.draw_scene(self.renderer, grid=True)
        self.npc.draw(self.renderer)          # робот II
        self.tram.draw(self.renderer)
        # лучи лидаров — только в главной 3D-сцене
        self.tram.draw_sensors(self.renderer, self.lidar_top, self.lidar_bot)
        self.renderer.end_view()

        if not direct:
            self.screen.blit(pygame.transform.smoothscale(
                scene, self.screen.get_size()), (0, 0))
        k = self.w / target.get_width()

        # маркеры зон A/B (текст поверх 3D)
        for name, (zx, zz), col in (("A", self.world.zones["A"], config.PAL["pad_a"]),
                                    ("B", self.world.zones["B"], config.PAL["pad_b"])):
            p = self.renderer.project_last((zx, 3.1, zz))
            if p:
                sx, sy = p[0] * k, p[1] * k
                f = self.font(17, bold=True)
                tag = f.render(name, True, (16, 20, 26))
                tw = tag.get_width()
                pygame.draw.circle(self.screen, col, (int(sx), int(sy)), 12)
                self.screen.blit(tag, (sx - tw / 2, sy - 10))

        self.wm.draw(self.screen)
        self._bars()
        self._toasts()

        # FPS
        self.frame_ms = (time.perf_counter() - t0) * 1000.0
        inst = 1000.0 / max(0.01, self.frame_ms)
        self.fps = self.fps * 0.9 + inst * 0.1
        self._fps_accum.append(inst)
        if len(self._fps_accum) > 60:
            self._fps_accum.pop(0)
            self.fps_avg = sum(self._fps_accum) / len(self._fps_accum)
            self.fps_min = min(self._fps_accum)

    def _chip(self, text, color, fnt, size):
        """Кэш чипа верхней панели (текст меняется редко → рендер раз в смену)."""
        key = (text, color, size)
        surf = getattr(self, "_chip_cache", None)
        if surf is None:
            surf = self._chip_cache = {}
        srf = surf.get(key)
        if srf is None:
            srf = fnt.render(text, True, color)
            surf[key] = srf
            if len(surf) > 24:
                surf.clear()
                srf = fnt.render(text, True, color)
                surf[key] = srf
        return srf

    def _bars(self):
        s = self.screen
        w = self.w
        # верхняя панель
        pygame.draw.rect(s, (12, 15, 22, 235), (0, 0, w, 30))
        pygame.draw.line(s, (60, 76, 110), (0, 30), (w, 30))
        if not hasattr(self, "_title_surf"):
            title = (f"ТРАМВАЙЧИК 3D v{config.APP_VERSION} · 2 склада "
                     f"{config.WAREHOUSE_W:.0f}×{config.WAREHOUSE_D:.0f} м · "
                     f"маршрут {config.ROUTE_LEN:.0f} м · "
                     f"тягач {config.LOCO_LENGTH:.1f}×{config.LOCO_WIDTH:.1f} м + "
                     f"{self.tram.n_wagons} вагона")
            self._title_surf = self.font(14, bold=True).render(title, True, TEXT)
        s.blit(self._title_surf, (12, 7))
        tx = 12 + self._title_surf.get_width() + 26
        score_col = OK if self.world.score >= 0 else WARN
        s.blit(self.font(15, bold=True).render(
            f"СЧЁТ {self.world.score:+d}", True, score_col), (tx, 6))
        s.blit(self.font(12).render(
            f"доставок: {self.tram.cycles} · наездов: {self.world.hits} · "
            f"АКБ {self.tram.battery:.0f}%", True, DIM),
            (tx + 128, 9))
        mode_txt = "АВТОПИЛОТ" if self.tram.auto else "РУЧНОЙ"
        mode_col = OK if self.tram.auto else ACC
        f = self.font(13, bold=True)
        mw = f.size(mode_txt)[0] + 16
        pygame.draw.rect(s, (24, 30, 44), (w / 2 - 130, 4, 130, 22), border_radius=11)
        s.blit(self._chip(mode_txt, mode_col, f, 13), (w / 2 - 122, 8))
        st_txt = self.tram.state
        pygame.draw.rect(s, (24, 30, 44), (w / 2 + 10, 4, 180, 22), border_radius=11)
        s.blit(self._chip(st_txt, TEXT, f, 13), (w / 2 + 20, 8))
        if self.tram.warn:
            pygame.draw.rect(s, (40, 20, 20), (w / 2 + 200, 4, 190, 22), border_radius=11)
            pygame.draw.rect(s, WARN, (w / 2 + 200, 4, 190, 22), width=2, border_radius=11)
            s.blit(f.render("ПЕШЕХОД НА ПУТИ", True, WARN), (w / 2 + 210, 8))
        if getattr(self.tram, "reverse_hold", 0.0) > 0.0:
            rx = w / 2 + 400
            pygame.draw.rect(s, (40, 32, 14), (rx, 4, 190, 22), border_radius=11)
            pygame.draw.rect(s, (255, 200, 80), (rx, 4, 190, 22), width=2,
                             border_radius=11)
            s.blit(f.render("РЕВЕРС ЗАПРЕЩЁН", True, (255, 200, 80)), (rx + 12, 8))
        if self.world.break_active:
            bx = w / 2 + 400
            pygame.draw.rect(s, (46, 36, 12), (bx, 4, 200, 22), border_radius=11)
            pygame.draw.rect(s, (255, 200, 80), (bx, 4, 200, 22), width=2,
                             border_radius=11)
            s.blit(f.render("ОБЕДЕННЫЙ ПЕРЕРЫВ", True, (255, 200, 80)),
                   (bx + 14, 8))
        # ускорение времени (v1.6.0) и вид камеры
        ts_txt = f"ВРЕМЯ ×{self.time_scale}" if self.time_scale > 1 else "ВРЕМЯ ×1"
        ts_col = (120, 220, 255) if self.time_scale > 1 else DIM
        pygame.draw.rect(s, (24, 30, 44), (w / 2 - 296, 4, 110, 22),
                         border_radius=11)
        s.blit(self.font(12, bold=True).render(ts_txt, True, ts_col),
               (w / 2 - 288, 8))
        view_txt = {"orbit": "ОРБИТА", "chase": "СЛЕДЯЩАЯ",
                    "cab": "1-е ЛИЦО"}[self.view_mode]
        pygame.draw.rect(s, (24, 30, 44), (w / 2 - 178, 4, 62, 22),
                         border_radius=11)
        s.blit(self.font(11, bold=True).render(view_txt, True, (150, 200, 255)),
               (w / 2 - 172, 9))
        fps_col = OK if self.fps >= 60 else (240, 200, 80)
        s.blit(self.font(14, bold=True).render(f"{self.fps:5.1f} FPS", True, fps_col),
               (w - 96, 7))
        s.blit(self.font(11).render("F1 — окна", True, DIM), (w - 200, 10))
        # нижняя панель
        pygame.draw.rect(s, (12, 15, 22, 235), (0, self.h - 26, w, 26))
        if not hasattr(self, "_hints_surf"):
            hints = ("TAB режим · WASD езда (реверс запрещён) · T время ×1-×8 · "
                     "F вид из кабины · ЛКМ облёт · колесо зум · M камера · "
                     "F1 справка · F6 упорядочить · O запись · R сброс · ESC")
            self._hints_surf = self.font(12).render(hints, True, DIM)
        s.blit(self._hints_surf, (10, self.h - 20))
        # штрафные очки справа (v1.3.0)
        wd = self.world
        cnt = (f"доставки {self.tram.cycles} · наезды {wd.hits} · "
               f"стены {wd.wall_hits} · груз {wd.cargo_wall_hits + wd.cargo_ped_hits}")
        fc = self.font(11)
        cw = fc.size(cnt)[0]
        s.blit(fc.render(cnt, True, (218, 168, 128)), (w - cw - 12, self.h - 19))

    def _toasts(self):
        y = self.h - 60
        for (txt, ttl, col) in reversed(self.toasts):
            a = min(1.0, ttl / 0.5)
            f = self.font(13, bold=True)
            surf = pygame.Surface((f.size(txt)[0] + 20, 24), pygame.SRCALPHA)
            surf.fill((20, 26, 40, int(230 * a)))
            surf.blit(f.render(txt, True, col), (10, 3))
            self.screen.blit(surf, (self.w / 2 - surf.get_width() / 2, y))
            y -= 28

    # -------------------------------------------------------------- самотест
    def _selftest(self):
        import time as _t
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        # тестовый режим таймингов: операции ускорены, чтобы за фазу замера
        # состав успел выехать из зоны A (реальные тайминги — в config.py)
        config.STOP_LINE_TIME = 0.5
        config.LOAD_TIME = 3.0
        config.UNLOAD_TIME = 3.0
        self.tram.timer = min(self.tram.timer, 0.5)   # не ждать полный стоп-линию
        print("САМОТЕСТ (headless): 2 фазы по 420 кадров, автопилот A→B")
        print(f"мир: 2 склада {config.WAREHOUSE_W:.0f}×{config.WAREHOUSE_D:.0f} м "
              f"H {config.WALL_H:.1f} м · маршрут {config.ROUTE_LEN:.0f} м "
              f"({config.ROUTE_TURNS} поворота) · малый круг R "
              f"{config.LOOP_R:.1f} м")
        print(f"состав: {self.tram.length_total():.1f} м · "
              f"масса {self.tram.mass():.0f} кг · 2 штабелёра")
        print(f"сенсоры: {self.lidar_top.rays}+{self.lidar_bot.rays} лучей, "
              f"камеры {self.onboard.w}x{self.onboard.h} + {self.rear.w}x{self.rear.h}, "
              f"робот II на маршруте")
        self.keys = {"up": False, "down": False, "left": False, "right": False,
                     "brake": False}

        def bench(frames, tag, scale=1):
            self.time_scale = scale
            times = []
            start_pos = self.tram.pos.copy()
            for i in range(frames):
                t0 = _t.perf_counter()
                self.update(1.0 / 60.0)
                self.render()
                times.append((_t.perf_counter() - t0) * 1000.0)
                if i == frames - 20:
                    try:
                        pygame.image.save(self.screen, os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "screenshot_selftest.png"))
                    except Exception:  # noqa: BLE001
                        pass
            avg_ms = sum(times) / len(times)
            moved = float(np.hypot(*(self.tram.pos - start_pos)))
            print(f"[{tag}] средний кадр: {avg_ms:.2f} мс | ≈FPS: {1000 / avg_ms:.0f} | "
                  f"граней: {self.renderer.stats['faces']} | "
                  f"пройдено: {moved:.1f} м (время ×{scale})")
            return 1000.0 / avg_ms, moved

        print(f"раскладка v1.6.0: окон {len(self.wm.wins)}, "
              f"активных по умолчанию "
              f"{sum(1 for wv in self.wm.wins if wv.visible)} "
              f"(YOLO — только №1, сетка по числу открытых)")
        # фаза 1 — производственная конфигурация (окна по умолчанию, ×1)
        fps_prod, moved1 = bench(420, "prod: окна по умолчанию", 1)
        # фаза 2 — стресс: все окна включены + ускорение времени ×4
        for wv in self.wm.wins:
            wv.visible = True
        panels.layout_all(self, force=True)
        fps_stress, moved2 = bench(420, "stress: все окна + время ×4", 4)
        self.time_scale = 1
        print(f"состояние миссии: {self.tram.state} | у площадки {self.tram.at} | "
              f"рейсов: {self.tram.cycles} | АКБ: {self.tram.battery:.0f}% | "
              f"робот II: {self.npc.speed:.1f} м/с")
        wd = self.world
        print(f"штабелёры: A {wd.stacker_a.state} · B {wd.stacker_b.state}")
        print(f"штрафы: наезды {wd.hits} · стены {wd.wall_hits} · "
              f"груз-стены {wd.cargo_wall_hits} · груз-люди {wd.cargo_ped_hits} · "
              f"робот II {wd.npc_hits} · счёт {wd.score:+d}")
        print(f"наблюдения: перерывы {wd.breaks_count} · потери цикла "
              f"{wd.break_loss} с · синих тележек выкачено {len(wd.carts)}")
        print("восприятие: " +
              ", ".join(f"{k}={len(v)}" for k, v in (self.onboard.outputs or {}).items()))
        lt, lb = self.lidar_top, self.lidar_bot
        print(f"лидары: верхний {len(lt.scan)}/{lt.rays} лучей "
              f"({lt.nearest()[0]:.1f} м {config.LIDAR_KIND_RU[lt.nearest()[1]]}), "
              f"нижний {len(lb.scan)}/{lb.rays} "
              f"({lb.nearest()[0]:.1f} м {config.LIDAR_KIND_RU[lb.nearest()[1]]})")
        sensors_ok = (len(lt.scan) == lt.rays and len(lb.scan) == lb.rays and
                      lb.nearest()[0] <= lb.range)   # верхний в коридоре может быть чист (18 м)
        # состав должен реально двигаться (полные рейсы проверяют test)
        # пороги откалиброваны по headless-песочнице (≈ на 20-25% медленнее
        # рабочей машины: здесь даёт ~50 FPS, на рабочей — 60+)
        ok = (fps_prod >= 38 and fps_stress >= 26 and
              (moved1 + moved2) > 4.0 and sensors_ok)
        pygame.quit()
        print("РЕЗУЛЬТАТ:", "OK" if ok else "FAIL")
        return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Трамвайчик 3D — два склада и маршрут 150 м")
    ap.add_argument("--selftest", action="store_true", help="headless-самопроверка + FPS")
    ap.add_argument("--cam", type=int, default=None, help="индекс веб-камеры")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    args = ap.parse_args()
    sys.exit(App(args).run())


if __name__ == "__main__":
    main()
