# -*- coding: utf-8 -*-
"""Программный 3D-рендерер «Трамвайчик 3D».

Возможности:
- алгоритм художника (сортировка граней и линий по глубине);
- отсечение задних граней по мировой нормали;
- клиппинг полигонов по near-плоскости (Sutherland–Hodgman, 1 плоскость);
- плоское освещение (ambient + diffuse);
- несколько вьюпортов за кадр (главный вид + бортовая камера) без пересоздания.

Оптимизации под 60+ FPS: все преобразования вершин — векторизованный numpy;
горячий цикл по граням работает на списках float (без numpy-объектов на грань).
"""
import math
import time
import numpy as np
import pygame

NEAR = 0.08
CULL_DIST = 42.0     # дальность отрисовки статики, м (дальше — невидима)


def _norm3(v):
    l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / l, v[1] / l, v[2] / l)


# Направление НА источник света (мировые координаты)
LIGHT = _norm3((0.42, 0.85, -0.34))
# Заполняющий свет с противоположной стороны — объёмность граней
LIGHT_FILL = _norm3((-0.46, 0.16, 0.78))


def shade(nx, ny, nz):
    """Плоское освещение: ambient + диффуз + заполняющий свет.
    Даёт граням разную яркость по нормалям — объекты выглядят ОБЪЁМНЫМИ."""
    l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    diff = (nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]) / l
    fill = (nx * LIGHT_FILL[0] + ny * LIGHT_FILL[1] + nz * LIGHT_FILL[2]) / l
    sh = 0.33 + 0.55 * (diff if diff > 0.0 else 0.0) + \
        0.24 * (fill if fill > 0.0 else 0.0)
    return min(sh, 1.15)


def _clip_near(poly):
    """Отсечение полигона в пространстве вида по плоскости z = NEAR."""
    out = []
    n = len(poly)
    for i in range(n):
        ax, ay, az = poly[i]
        bx, by, bz = poly[(i + 1) % n]
        ain = az >= NEAR
        bin_ = bz >= NEAR
        if ain:
            out.append((ax, ay, az))
        if ain != bin_:
            t = (NEAR - az) / (bz - az)
            out.append((ax + (bx - ax) * t, ay + (by - ay) * t, NEAR))
    return out


class Renderer:
    def __init__(self):
        self.queue = []          # ("p", depth, pts2d, color) | ("l", depth, xyxy, color, width)
        self.cam = None          # активный CamData
        self.stats = {"faces": 0, "lines": 0, "ms": 0.0}

    # ------------------------------------------------------------------ вид
    def begin_view(self, surface, camdata, cx=None, cy=None, clip=None):
        self.cam = camdata
        self.queue = []
        camdata.surface = surface
        camdata.vw, camdata.vh = surface.get_size()
        camdata.cx = camdata.vw * 0.5 if cx is None else cx
        camdata.cy = camdata.vh * 0.5 if cy is None else cy
        camdata.clip = clip

    def project_last(self, p):
        """Мировая точка → (x, y, глубина) в параметрах последнего begin_view, или None."""
        c = self.cam
        if c is None:
            return None
        dx = p[0] - c.pos[0]
        dy = p[1] - c.pos[1]
        dz = p[2] - c.pos[2]
        x = dx * c.u[0] + dy * c.u[1] + dz * c.u[2]
        y = dx * c.v[0] + dy * c.v[1] + dz * c.v[2]
        z = dx * c.w[0] + dy * c.w[1] + dz * c.w[2]
        if z <= NEAR:
            return None
        return (c.cx + c.focal * x / z, c.cy - c.focal * y / z, z)

    # -------------------------------------------------------------- примитивы
    def draw_lines(self, segs, color, width=1, y=0.02):
        """Отрезки (x1, z1, x2, z2) на высоте y — сетка пола, границы зон и т.п.
        Векторизованная проекция (v1.5.0) — важно для лидаров/разметки."""
        n = len(segs)
        if n == 0:
            return
        c = self.cam
        if n >= 6:
            arr = np.asarray(segs, dtype=np.float64)
            P = np.empty((n * 2, 3))
            P[0::2, 0] = arr[:, 0]
            P[0::2, 1] = y
            P[0::2, 2] = arr[:, 1]
            P[1::2, 0] = arr[:, 2]
            P[1::2, 1] = y
            P[1::2, 2] = arr[:, 3]
            d = P - c.pos
            X = d @ c.u
            Yv = d @ c.v
            Z = d @ c.w
            ok = Z > NEAR
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = c.focal / np.where(ok, Z, 1.0)
            sx = c.cx + X * inv
            sy = c.cy - Yv * inv
            good = ok[0::2] & ok[1::2]
            sx_f = sx.tolist(); sy_f = sy.tolist(); z_f = Z.tolist()
            g_l = good.tolist()
            push = self.queue.append
            for i in range(n):
                if not g_l[i]:
                    continue
                push(("l", (z_f[2 * i] + z_f[2 * i + 1]) * 0.5,
                      (sx_f[2 * i], sy_f[2 * i], sx_f[2 * i + 1],
                       sy_f[2 * i + 1]), color, width))
            return
        for (x1, z1, x2, z2) in segs:
            p1 = self.project_last((x1, y, z1))
            p2 = self.project_last((x2, y, z2))
            if p1 is None or p2 is None:
                continue
            self.queue.append(("l", (p1[2] + p2[2]) * 0.5,
                               (p1[0], p1[1], p2[0], p2[1]), color, width))

    def draw_mesh(self, mesh, pos=(0.0, 0.0, 0.0), yaw=0.0):
        """Меш с поворотом вокруг Y и трансляцией. Горячий путь рендерера.

        Малые меши (≤ 24 вершин) — чистый Python без numpy (нет аллокаций);
        большие (слитая статика) — векторизованный numpy.
        Отсечения: ЦЕЛИК ПО BBOX МЕША, задняя грань, near-плоскость,
        вне экрана, площадь < 2 px.
        """
        c = self.cam
        if c is None:
            return

        # ---- отсечение целого меша по его габариту (8 углов bbox) ----
        bx0, by0, bz0, bx1, by1, bz1 = mesh.bbox
        ca0, sa0 = math.cos(yaw), math.sin(yaw)
        px_, py_, pz_ = pos
        zmax = -1e18
        zmin = 1e18
        minx = miny = -1e18
        maxx = maxy = 1e18
        crossed_near = False
        cpx0 = float(c.pos[0]); cpy0 = float(c.pos[1]); cpz0 = float(c.pos[2])
        ux0, uy0, uz0 = c.u; vx0, vy0, vz0 = c.v; wx0, wy0, wz0 = c.w
        focal0 = c.focal
        for cx_ in (bx0, bx1):
            for cy_ in (by0, by1):
                for cz_ in (bz0, bz1):
                    X = cx_ * ca0 + cz_ * sa0 + px_
                    Y = cy_ + py_
                    Z = -cx_ * sa0 + cz_ * ca0 + pz_
                    dx = X - cpx0; dy = Y - cpy0; dz = Z - cpz0
                    Zv = dx * wx0 + dy * wy0 + dz * wz0
                    if Zv < zmin:
                        zmin = Zv
                    if Zv > zmax:
                        zmax = Zv
                    if Zv < NEAR:
                        crossed_near = True
        if zmax < NEAR:
            return                        # весь меш позади камеры
        if zmin > CULL_DIST:
            return                        # слишком далеко
        if not crossed_near:
            # всё перед камерой → можно проецировать и отбросить по экрану
            VW0, VH0 = c.vw, c.vh
            cx0 = c.cx; cy0 = c.cy
            sminx = sminy = 1e18
            smaxx = smaxy = -1e18
            for cx_ in (bx0, bx1):
                for cy_ in (by0, by1):
                    for cz_ in (bz0, bz1):
                        X = cx_ * ca0 + cz_ * sa0 + px_
                        Y = cy_ + py_
                        Z = -cx_ * sa0 + cz_ * ca0 + pz_
                        dx = X - cpx0; dy = Y - cpy0; dz = Z - cpz0
                        Xv = dx * ux0 + dy * uy0 + dz * uz0
                        Yv = dx * vx0 + dy * vy0 + dz * vz0
                        Zv = dx * wx0 + dy * wy0 + dz * wz0
                        inv = focal0 / Zv
                        sx = cx0 + Xv * inv
                        sy = cy0 - Yv * inv
                        if sx < sminx: sminx = sx
                        if sx > smaxx: smaxx = sx
                        if sy < sminy: sminy = sy
                        if sy > smaxy: smaxy = sy
            if smaxx < 0 or smaxy < 0 or sminx > VW0 or sminy > VH0:
                return                    # весь меш вне экрана

        ca, sa = ca0, sa0
        cpx, cpy, cpz = float(c.pos[0]), float(c.pos[1]), float(c.pos[2])
        u, vv2, w2 = c.u, c.v, c.w
        ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
        vx, vy, vz = float(vv2[0]), float(vv2[1]), float(vv2[2])
        wx, wy, wz = float(w2[0]), float(w2[1]), float(w2[2])
        cx, cy, focal = c.cx, c.cy, c.focal
        VW, VH = c.vw, c.vh
        lx, ly, lz = LIGHT
        q = self.queue
        push = q.append

        if mesh.n_faces and len(mesh.vlist) <= 24:
            # ---- быстрый путь: чистый Python (малые меши) ----
            xwl = []; ywl = []; zwl = []
            xvl = []; yvl = []; zvl = []
            for (x, y, z) in mesh.vlist:
                X = x * ca + z * sa + pos[0]
                Y = y + pos[1]
                Z = -x * sa + z * ca + pos[2]
                xwl.append(X); ywl.append(Y); zwl.append(Z)
                dx = X - cpx; dy = Y - cpy; dz = Z - cpz
                xvl.append(dx * ux + dy * uy + dz * uz)
                yvl.append(dx * vx + dy * vy + dz * vz)
                zvl.append(dx * wx + dy * wy + dz * wz)
            self._faces_loop(mesh, xwl, ywl, zwl, xvl, yvl, zvl,
                             cpx, cpy, cpz, cx, cy, focal, VW, VH,
                             lx, ly, lz, push, ca, sa, pos[0], pos[1], pos[2])
            return

        # ---- numpy-путь (слитая статика и крупные динаминые меши) ----
        self._draw_mesh_vec(mesh, pos, ca, sa)

    # ------------------------------------------------- векторизованный путь
    def _face_arrays(self, mesh):
        """Кэш массивов граней: индексы (паддинг -1), нормали, центры, цвета."""
        if mesh._farr is None:
            nf = mesh.n_faces
            V = max(len(f[0]) for f in mesh.faces)
            fidx = np.full((nf, V), -1, dtype=np.int64)
            fn = np.empty((nf, 3))
            fc = np.empty((nf, 3))
            col = np.empty((nf, 3), dtype=np.int64)
            nocull = np.zeros(nf, dtype=bool)
            for i, (idxs, base, nc) in enumerate(mesh.faces):
                fidx[i, :len(idxs)] = idxs
                fn[i] = mesh.fnorm[i]
                fc[i] = mesh.fcent[i]
                col[i] = base
                nocull[i] = nc
            mesh._farr = (fidx, fn, fc, col, nocull)
        return mesh._farr

    def _draw_mesh_vec(self, mesh, pos, ca, sa):
        """Векторизованный путь: трансформация вершин, backface/near/экранные
        отсечения по массивам; Python-цикл только по выжившим граням."""
        c = self.cam
        v = mesh.verts
        xw = (v[:, 0] * ca + v[:, 2] * sa + pos[0])
        yw = (v[:, 1] + pos[1])
        zw = (-v[:, 0] * sa + v[:, 2] * ca + pos[2])
        dx = xw - c.pos[0]
        dy = yw - c.pos[1]
        dz = zw - c.pos[2]
        xv = dx * c.u[0] + dy * c.u[1] + dz * c.u[2]
        yv = dx * c.v[0] + dy * c.v[1] + dz * c.v[2]
        zv = dx * c.w[0] + dy * c.w[1] + dz * c.w[2]

        fidx, fn, fc, col, nocull = self._face_arrays(mesh)
        valid = fidx >= 0
        idx2 = np.where(valid, fidx, 0)
        # --- отсечение задних граней (векторизовано; нормаль К камере) ---
        if mesh.static:
            d0 = c.pos[0] - fc[:, 0]
            d1 = c.pos[1] - fc[:, 1]
            d2 = c.pos[2] - fc[:, 2]
            dot = fn[:, 0] * d0 + fn[:, 1] * d1 + fn[:, 2] * d2
        else:
            nx = fn[:, 0] * ca + fn[:, 2] * sa
            nz = -fn[:, 0] * sa + fn[:, 2] * ca
            ny = fn[:, 1]
            mx = fc[:, 0] * ca + fc[:, 2] * sa + pos[0]
            my = fc[:, 1] + pos[1]
            mz = -fc[:, 0] * sa + fc[:, 2] * ca + pos[2]
            dot = nx * (c.pos[0] - mx) + ny * (c.pos[1] - my) + \
                nz * (c.pos[2] - mz)
        alive = (dot > 0.0) | nocull          # видима грань, чья нормаль смотрит НА камеру
        if not alive.any():
            return
        # --- near-отсечение по вершинам ---
        zv_f = zv[idx2]
        behind = (zv_f < NEAR) | ~valid
        all_behind = behind.all(axis=1)
        alive &= ~all_behind
        if not alive.any():
            return
        any_near = ((zv_f < NEAR) & valid).any(axis=1)
        # --- экранные координаты всех вершин (векторизовано) ---
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = c.focal / zv_f
        sx = c.cx + xv[idx2] * inv
        sy = c.cy - yv[idx2] * inv
        sx = np.where(valid, sx, 1e18)
        sy = np.where(valid, sy, 1e18)
        # вершины за near — не проецируются корректно: грань пойдёт в клиппер
        bad = (zv_f < NEAR) & valid
        sx = np.where(bad, np.nan, sx)
        sy = np.where(bad, np.nan, sy)
        minx = np.nanmin(np.where(np.isnan(sx), np.inf, sx), axis=1)
        maxx = np.nanmax(np.where(np.isnan(sx), -np.inf, sx), axis=1)
        miny = np.nanmin(np.where(np.isnan(sy), np.inf, sy), axis=1)
        maxy = np.nanmax(np.where(np.isnan(sy), -np.inf, sy), axis=1)
        onscreen = ~np.isnan(sx).any(axis=1)          # без вершин за near
        alive &= onscreen & ~(maxx < 0) & ~(maxy < 0) & \
            ~(minx > c.vw) & ~(miny > c.vh)
        area = (maxx - minx) * (maxy - miny)
        alive &= area >= 2.0
        if not alive.any():
            return
        # --- освещение (динамика) — векторизовано ---
        if mesh.static:
            cols = col
        else:
            nx = fn[:, 0] * ca + fn[:, 2] * sa
            nz = -fn[:, 0] * sa + fn[:, 2] * ca
            ny = fn[:, 1]
            ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
            diff = (nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]) / ln
            fill = (nx * LIGHT_FILL[0] + ny * LIGHT_FILL[1] + nz * LIGHT_FILL[2]) / ln
            sh = 0.33 + 0.55 * np.clip(diff, 0.0, 1.0) + \
                0.24 * np.clip(fill, 0.0, 1.0)
            cols = np.clip((col * sh[:, None]).astype(np.int64), 0, 255)
        # --- Python-цикл только по видимым граням ---
        depth_v = zv[idx2]                       # (nf, V)
        depth_v = np.where(valid, depth_v, 0.0)
        cnt = valid.sum(axis=1).astype(np.float64)
        depth = (depth_v.sum(axis=1) / np.maximum(cnt, 1.0))
        xvl = xv.tolist(); yvl = yv.tolist(); zvl = zv.tolist()
        fidx_l = fidx.tolist()
        valid_l = valid.tolist()
        cols_l = cols.tolist()
        alive_l = alive.tolist()
        depth_l = depth.tolist()
        q = self.queue
        push = q.append
        camx, camy, camz = c.pos[0], c.pos[1], c.pos[2]
        for fi in range(mesh.n_faces):
            if not alive_l[fi]:
                continue
            scr = []
            vidx = fidx_l[fi]
            vok = valid_l[fi]
            for k, vi in enumerate(vidx):
                if not vok[k]:
                    continue
                Z = zvl[vi]
                if Z < NEAR:                     # страховка (клиппер выше)
                    scr = None
                    break
                inv = c.focal / Z
                scr.append((int(c.cx + xvl[vi] * inv),
                            int(c.cy - yvl[vi] * inv)))
            if not scr:
                continue
            push(("p", depth_l[fi], scr, tuple(cols_l[fi])))

    def _faces_loop(self, mesh, xwl, ywl, zwl, xv, yv, zv,
                    cpx, cpy, cpz, cx, cy, focal, VW, VH,
                    lx, ly, lz, push, ca=1.0, sa=0.0, px=0.0, py=0.0, pz=0.0):
        """Общий цикл по граням. Нормали/центры берутся из кэша меша;
        для динамики поворачиваются (дешевле векторного произведения)."""
        is_static = mesh.static
        for fi, (idxs, base, no_cull) in enumerate(mesh.faces):
            if is_static:
                nx, ny, nz = mesh.fnorm[fi]
                if not no_cull:
                    mx, my, mz = mesh.fcent[fi]
                    # грань видима, когда нормаль смотрит НА камеру
                    if nx * (cpx - mx) + ny * (cpy - my) + nz * (cpz - mz) <= 0.0:
                        continue
            else:
                nx0, ny0, nz0 = mesh.fnorm[fi]
                nx = nx0 * ca + nz0 * sa
                nz = -nx0 * sa + nz0 * ca
                ny = ny0
                if not no_cull:
                    mx0, my0, mz0 = mesh.fcent[fi]
                    mx = mx0 * ca + mz0 * sa + px
                    my = my0 + py
                    mz = -mx0 * sa + mz0 * ca + pz
                    if nx * (cpx - mx) + ny * (cpy - my) + nz * (cpz - mz) <= 0.0:
                        continue

            vv = []
            all_behind = True
            for i in idxs:
                Z = zv[i]
                if Z >= NEAR:
                    all_behind = False
                vv.append((xv[i], yv[i], Z))
            if all_behind:
                continue
            if any(Z < NEAR for (_, _, Z) in vv):
                vv = _clip_near(vv)
                if len(vv) < 3:
                    continue

            depth = 0.0
            minx = miny = 1e9
            maxx = maxy = -1e9
            scr = []
            for (X, Y, Z) in vv:
                inv = focal / Z
                sx = cx + X * inv
                sy = cy - Y * inv
                scr.append((int(sx), int(sy)))
                depth += Z
                if sx < minx: minx = sx
                if sx > maxx: maxx = sx
                if sy < miny: miny = sy
                if sy > maxy: maxy = sy
            depth /= len(vv)
            # отсечение по экрану и по площади (мелкие дальние грани)
            if maxx < 0 or maxy < 0 or minx > VW or miny > VH:
                continue
            if (maxx - minx) * (maxy - miny) < 2.0:
                continue

            if is_static:
                col = base  # цвет уже зашейковлен при сборке статики
            else:
                ln = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
                diff = (nx * lx + ny * ly + nz * lz) / ln
                fill = (nx * LIGHT_FILL[0] + ny * LIGHT_FILL[1] +
                        nz * LIGHT_FILL[2]) / ln
                sh = 0.33 + 0.55 * (diff if diff > 0.0 else 0.0) + \
                    0.24 * (fill if fill > 0.0 else 0.0)
                sh = min(sh, 1.15)
                col = (min(255, int(base[0] * sh)), min(255, int(base[1] * sh)),
                       min(255, int(base[2] * sh)))
            push(("p", depth, scr, col))

    # ---------------------------------------------------------------- вывод
    def end_view(self):
        """Сортировка по глубине (дальний → ближний) и отрисовка в поверхность."""
        c = self.cam
        t0 = time.perf_counter()
        self.queue.sort(key=lambda it: -it[1])
        surf = c.surface
        clip = c.clip
        if clip is not None:
            surf.set_clip(clip)
        n_p = n_l = 0
        for it in self.queue:
            if it[0] == "p":
                pygame.draw.polygon(surf, it[3], it[2])
                n_p += 1
            else:
                _, _, (x1, y1, x2, y2), col, wd = it
                pygame.draw.line(surf, col, (x1, y1), (x2, y2), wd)
                n_l += 1
        if clip is not None:
            surf.set_clip(None)
        self.stats["faces"] = n_p
        self.stats["lines"] = n_l
        self.stats["ms"] = (time.perf_counter() - t0) * 1000.0
