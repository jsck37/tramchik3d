# -*- coding: utf-8 -*-
"""Параметрические меши 3D-движка: параллелепипед, цилиндр, плоский четырёхугольник.

Грань = (кортеж индексов вершин, (r, g, b), no_cull).
Порядок обхода вершин выбран так, что нормаль (по 3 первым точкам) направлена НАРУЖУ:
грань отбрасывается, если нормаль смотрит от камеры.
"""
import math
import numpy as np

from .renderer import shade


class Mesh:
    __slots__ = ("verts", "faces", "n_faces", "vlist", "fnorm", "fcent",
                 "static", "bbox", "_farr")

    def __init__(self, verts, faces):
        self.verts = np.asarray(verts, dtype=np.float64)
        self.vlist = self.verts.tolist()   # для быстрого пути без numpy
        # габарит меша (для отсечения целого меша до цикла по граням)
        if len(self.vlist):
            xs = [v[0] for v in self.vlist]
            ys = [v[1] for v in self.vlist]
            zs = [v[2] for v in self.vlist]
            self.bbox = (min(xs), min(ys), min(zs),
                         max(xs), max(ys), max(zs))
        else:
            self.bbox = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._farr = None                  # кэш массивов граней (лендинг рендера)
        self.faces = []
        self.fnorm = []   # локальные нормали граней (не пересчитываем каждый кадр)
        self.fcent = []   # локальные центры граней
        for face in faces:
            idxs, color = face[0], face[1]
            no_cull = bool(face[2]) if len(face) > 2 else False
            idxs = tuple(idxs)
            self.faces.append((idxs, color, no_cull))
            p0 = self.vlist[idxs[0]]
            p1 = self.vlist[idxs[1]]
            p2 = self.vlist[idxs[2]]
            e1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
            e2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
            nx = e1[1] * e2[2] - e1[2] * e2[1]
            ny = e1[2] * e2[0] - e1[0] * e2[2]
            nz = e1[0] * e2[1] - e1[1] * e2[0]
            self.fnorm.append((nx, ny, nz))
            mx = my = mz = 0.0
            for i in idxs:
                p = self.vlist[i]
                mx += p[0]; my += p[1]; mz += p[2]
            k = 1.0 / len(idxs)
            self.fcent.append((mx * k, my * k, mz * k))
        self.n_faces = len(self.faces)
        self.static = False   # True — нормали/цвет уже в мировых координатах


def merge_meshes(parts):
    """Слияние списка (mesh, (x,y,z), yaw) в один меш в мировых координатах.

    Применяется к статике: один draw_mesh вместо сотен; нормали, центры и
    ЗАРАНЕЕ ЗАШЕЙКОВЛЕННЫЙ цвет вычисляются один раз при сборке.
    """
    all_v = []
    all_f = []
    for mesh, pos, yaw in parts:
        base = len(all_v)
        ca, sa = math.cos(yaw), math.sin(yaw)
        for (x, y, z) in mesh.vlist:
            all_v.append((x * ca + z * sa + pos[0], y + pos[1],
                          -x * sa + z * ca + pos[2]))
        for (idxs, color, no_cull) in mesh.faces:
            all_f.append((tuple(i + base for i in idxs), color, no_cull))
    m = Mesh(all_v, all_f)
    m.static = True
    # мировой свет фиксирован → цвет каждой грани считаем один раз
    # (двойной свет — объёмная картинка)
    lit = []
    for (idxs, color, no_cull), (nx, ny, nz) in zip(m.faces, m.fnorm):
        sh = shade(nx, ny, nz)
        lit.append((idxs, (min(255, int(color[0] * sh)),
                           min(255, int(color[1] * sh)),
                           min(255, int(color[2] * sh))), no_cull))
    m.faces = lit
    return m


def merge_parts(parts):
    """Слияние мешей в ЛОКАЛЬНЫХ координатах без запекания света (для динамики).

    Возвращает обычный Mesh: нормали и освещение считаются при отрисовке
    (меши умеют вращаться). Используется, чтобы рисовать составные объекты
    (паровозик, прицепы) одним draw_mesh вместо десятков."""
    all_v = []
    all_f = []
    for mesh, pos, yaw in parts:
        base = len(all_v)
        ca, sa = math.cos(yaw), math.sin(yaw)
        for (x, y, z) in mesh.vlist:
            all_v.append((x * ca + z * sa + pos[0], y + pos[1],
                          -x * sa + z * ca + pos[2]))
        for (idxs, color, no_cull) in mesh.faces:
            all_f.append((tuple(i + base for i in idxs), color, no_cull))
    return Mesh(all_v, all_f)


def box_mesh(sx, sy, sz, color, cx=0.0, cy=0.0, cz=0.0):
    """Параллелепипед размером (sx, sy, sz) с центром в (cx, cy, cz). 6 граней."""
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    v = [(cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
         (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
         (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
         (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz)]
    f = [((4, 5, 6, 7), color), ((1, 0, 3, 2), color), ((5, 1, 2, 6), color),
         ((0, 4, 7, 3), color), ((3, 7, 6, 2), color), ((0, 1, 5, 4), color)]
    return Mesh(v, f)


def quad_mesh(p0, p1, p2, p3, color, no_cull=False):
    """Плоский четырёхугольник из 4 точек (мировые координаты)."""
    return Mesh([p0, p1, p2, p3], [((0, 1, 2, 3), color, no_cull)])


def cylinder_mesh(radius, length, segments=12, color=(200, 200, 200), axis="x",
                  cx=0.0, cy=0.0, cz=0.0):
    """Цилиндр радиуса radius, длиной length вдоль оси axis ("x" | "y" | "z"),
    с центром в (cx, cy, cz). Торцы — с корректной намоткой (cull работает)."""
    half = length * 0.5
    verts = []
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        y, z = radius * math.cos(a), radius * math.sin(a)
        verts.append((half, y, z))    # 2i   — переднее кольцо (+X)
        verts.append((-half, y, z))   # 2i+1 — заднее кольцо (-X)
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        a0, b0, a1, b1 = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
        faces.append(((a0, b0, b1, a1), color, False))
    # торцы: обход подобран так, что нормаль смотрит наружу → backface-cull валиден
    ci = len(verts)
    verts.append((half, 0.0, 0.0))
    faces.append((tuple([ci] + [2 * i for i in range(segments)]), color, False))
    ci2 = len(verts)
    verts.append((-half, 0.0, 0.0))
    faces.append((tuple([ci2] + [2 * (segments - 1 - i) + 1 for i in range(segments)]),
                  color, False))
    # переориентация оси (собственные повороты, det = +1 → намотка сохраняется)
    if axis == "y":      # +X → +Y
        verts = [(-vy, vx, vz) for (vx, vy, vz) in verts]
    elif axis == "z":    # +X → +Z
        verts = [(-vz, vy, vx) for (vx, vy, vz) in verts]
    if cx or cy or cz:
        verts = [(vx + cx, vy + cy, vz + cz) for (vx, vy, vz) in verts]
    return Mesh(verts, faces)
