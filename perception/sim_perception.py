# -*- coding: utf-8 -*-
"""Перцепция в симуляции: 7 задач YOLO, синтезируемые из «истинных» данных мира.

Задачи: detect, pose, segment, classify, obb, semantic, depth.
Все выходные координаты — в системе бортовой камеры (ONBOARD_W × ONBOARD_H).
"""
import math
import numpy as np

import config
from engine.vecmath import clamp
from sim.pedestrians import LINKS

PAD = config.PAL
CLASSES_RU = {"person": "человек", "crate": "ящик", "cart": "тележка",
              "tram": "трамвайчик", "rack": "стеллаж"}
CLASS_COLOR = {"person": (90, 220, 130), "crate": (240, 200, 70),
               "cart": (90, 200, 240), "tram": (250, 140, 70), "rack": (150, 160, 190)}
# CID для экспорта в YOLO-датасет
CID = {"person": 0, "crate": 1, "rack": 1, "cart": 2, "tram": 3}


def convex_hull(points):
    """Выпуклая оболочка (алгоритм Эндрю, монотонная цепочка)."""
    pts = sorted(set((float(x), float(y)) for (x, y) in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def compute_outputs(world, tram, project, vw, vh, t):
    """Собирает выходы всех 7 задач. project(world_pt) → (x, y, глубина) | None."""
    out = {k: [] for k in config.TASKS}

    # ---------------------------------------------------------------- ящики/стеллажи/прицеп
    def box_object(label, corners3d, fwd2d=None):
        proj = [project(p) for p in corners3d]
        vis = [p for p in proj if p is not None]
        if len(vis) < 4:
            return
        xs = [p[0] for p in vis]
        ys = [p[1] for p in vis]
        x0, y0 = max(0, min(xs)), max(0, min(ys))
        x1, y1 = min(vw - 1, max(xs)), min(vh - 1, max(ys))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return
        d = min(p[2] for p in vis)
        conf = clamp(1.12 - d / 30.0, 0.35, 0.97)
        out["detect"].append({"label": label, "conf": conf,
                              "box": (x0, y0, x1 - x0, y1 - y0)})
        # OBB: угол — проекция направления «вперёд» объекта
        angle = 0.0
        if fwd2d is not None:
            c = ((x0 + x1) / 2, (y0 + y1) / 2, d)
            p0 = project(np.array([c[0], 0.5, c[1]]))
            p1 = project(np.array([c[0] + fwd2d[0] * 0.8, 0.5, c[1] + fwd2d[1] * 0.8]))
            if p0 and p1:
                angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        out["obb"].append({"label": label, "conf": conf, "angle": angle,
                           "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
                           "w": x1 - x0, "h": y1 - y0})
        hull = convex_hull([(p[0], p[1]) for p in vis])
        if len(hull) >= 3:
            cx = sum(p[0] for p in hull) / len(hull)
            cy = sum(p[1] for p in hull) / len(hull)
            hull = [(cx + (p[0] - cx) * 1.18, cy + (p[1] - cy) * 1.18) for p in hull]
            out["segment"].append({"label": label, "pts": hull})
        return d

    for cr in world.crates:
        x, z = cr["pos"]
        sx, sy, sz = cr["size"]
        lbl = "cart" if cr.get("kind") == "cart" else "crate"
        corners = [np.array([x + dx * sx / 2, ly, z + dz * sz / 2])
                   for dx in (-1, 1) for ly in (0.02, sy) for dz in (-1, 1)]
        box_object(lbl, corners)

    for rk in world.racks:
        x, z = rk["pos"]
        sx, sz = rk["size"]
        corners = [np.array([x + dx * sx / 2, ly, z + dz * sz / 2])
                   for dx in (-1, 1) for ly in (0.02, 3.0) for dz in (-1, 1)]
        box_object("rack", corners)

    # тягач — РЕАЛЬНЫЙ габарит 1,6 × 0,8 × 2,2 м
    loco_corners = tram.box_corners(tram.pos, 0.05, config.LOCO_HEIGHT,
                                    config.LOCO_WIDTH / 2 + 0.03,
                                    config.LOCO_LENGTH / 2 + 0.03, tram.yaw)
    box_object("tram", loco_corners, tram.fwd())

    # вагоны — тележки 0,85 × 0,7 м (следуют за тягачом)
    for wp, wy in zip(tram.wagon_pos, tram.wagon_yaw):
        cart_corners = tram.box_corners(wp, 0.05, 1.0,
                                        config.WAGON_WIDTH / 2 + 0.03,
                                        config.WAGON_LENGTH / 2 + 0.08, wy)
        box_object("cart", cart_corners, (math.sin(wy), math.cos(wy)))

    # робот II — второй состав: детекция как «tram»/«cart»
    npc = getattr(world, "npc", None)
    if npc is not None:
        nloco = npc.box_corners(npc.pos, 0.05, config.LOCO_HEIGHT,
                                config.LOCO_WIDTH / 2 + 0.03,
                                config.LOCO_LENGTH / 2 + 0.03, npc.yaw)
        box_object("tram", nloco, npc.fwd())
        for wp, wy in zip(npc.wagon_pos, npc.wagon_yaw):
            nc = npc.box_corners(wp, 0.05, 1.0,
                                 config.WAGON_WIDTH / 2 + 0.03,
                                 config.WAGON_LENGTH / 2 + 0.08, wy)
            box_object("cart", nc, (math.sin(wy), math.cos(wy)))

    # штабелёры — детекция как «cart»
    for st in getattr(world, "stackers", []):
        stc = tram.box_corners(st.pos, 0.05, 2.0, 0.48, 0.62, st.yaw)
        box_object("cart", stc, (math.sin(st.yaw), math.cos(st.yaw)))

    # ---------------------------------------------------------------- пешеходы (детекция/позы/сегментация)
    for ped in world.pedestrians:
        kps = ped.keypoints3d()
        proj = [project(p) for p in kps]
        vis = [p for p in proj if p is not None]
        if len(vis) < 4:
            continue
        d = min(p[2] for p in vis)
        conf = clamp(1.10 - d / 26.0, 0.4, 0.98)
        xs = [p[0] for p in vis]
        ys = [p[1] for p in vis]
        x0, y0 = max(0, min(xs)), max(0, min(ys))
        x1, y1 = min(vw - 1, max(xs)), min(vh - 1, max(ys))
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        out["detect"].append({"label": "person", "id": ped.id, "conf": conf,
                              "box": (x0, y0, x1 - x0, y1 - y0)})
        pts2 = [(p[0], p[1], 1) if p is not None else (0.0, 0.0, 0) for p in proj]
        out["pose"].append({"pts": pts2, "links": LINKS})
        hull = convex_hull([(p[0], p[1]) for p in vis])
        if len(hull) >= 3:
            cx = sum(p[0] for p in hull) / len(hull)
            cy = sum(p[1] for p in hull) / len(hull)
            hull = [(cx + (p[0] - cx) * 1.3, cy + (p[1] - cy) * 1.15) for p in hull]
            out["segment"].append({"label": "person", "pts": hull})
        p0 = project(np.array([ped.pos[0], 1.0, ped.pos[1]]))
        f = ped.fwd()
        p1 = project(np.array([ped.pos[0] + f[0] * 0.7, 1.0, ped.pos[1] + f[1] * 0.7]))
        angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0]) if (p0 and p1) else 0.0
        out["obb"].append({"label": "person", "conf": conf, "angle": angle,
                           "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
                           "w": (x1 - x0) * 0.72, "h": y1 - y0})

    # ---------------------------------------------------------------- classify (top-5)
    weights = {}
    for det in out["detect"]:
        lbl = det["label"]
        w = det["conf"] * (2.4 if lbl == "person" else 1.6)
        weights[lbl] = max(weights.get(lbl, 0.0), w)
    total = sum(weights.values())
    if total > 1e-6:
        rank = sorted(weights.items(), key=lambda kv: -kv[1])[:5]
        out["classify"] = [(CLASSES_RU.get(k, k), v / total) for k, v in rank]
    else:
        out["classify"] = [("пусто", 1.0)]

    # ---------------------------------------------------------------- semantic (зоны + следы объектов)
    zw, zd = config.ZONE_SIZE
    for name, (zx, zz), col in (("A", world.zones["A"], PAD["pad_a"]),
                                ("B", world.zones["B"], PAD["pad_b"])):
        poly = []
        for (dx, dz) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = project(np.array([zx + dx * zw / 2, 0.03, zz + dz * zd / 2]))
            if p:
                poly.append((p[0], p[1]))
        if len(poly) >= 3:
            out["semantic"].append({"label": "зона " + name, "col": col, "pts": poly})
    for det, seg in zip([d for d in out["detect"]], out["segment"]):
        col = CLASS_COLOR.get(det["label"], (200, 200, 200))
        out["semantic"].append({"label": det["label"], "col": col, "pts": seg["pts"]})

    # ---------------------------------------------------------------- depth (сетка пола + объекты)
    f = tram.fwd()
    tp = tram.pos
    for gx in range(-13, 14, 3):
        for gz in range(-8, 9, 2):
            wx, wz = tp[0] + gx, tp[1] + gz
            if not world.contains(wx, wz, 0.8):
                continue
            rel = np.array([wx - tp[0], wz - tp[1]])
            dist = float(np.hypot(rel[0], rel[1]))
            if dist < 0.5 or dist > 16:
                continue
            p = project(np.array([wx, 0.02, wz]))
            if p is None:
                continue
            gray = clamp(int(255 - dist * 9.2), 22, 235)
            size = clamp(int(30.0 / dist), 2, 12)
            out["depth"].append((p[0], p[1], size, gray))
    return out


def to_yolo_lines(outputs, vw, vh):
    """detect → строки YOLO-формата: cid cx cy w h (нормализовано)."""
    lines = []
    for det in outputs.get("detect", []):
        cid = CID.get(det["label"], 1)
        x, y, w, h = det["box"]
        cx = (x + w / 2) / vw
        cy = (y + h / 2) / vh
        lines.append(f"{cid} {cx:.5f} {cy:.5f} {min(w / vw, 1):.5f} {min(h / vh, 1):.5f}")
    return lines
