# -*- coding: utf-8 -*-
"""Навигация: сетка занятости + A* (8 соседей) + сглаживание пути по видимости.

мир — ПРЯМОУГОЛЬНИК (два склада + территория маршрута 150 м),
сетка задаётся границами (x0, z0, x1, z1), а не половинами размеров.
"""
import heapq
import math


class NavGrid:
    def __init__(self, x0, z0, x1, z1, cell=1.0):
        self.x0 = float(x0)
        self.z0 = float(z0)
        self.x1 = float(x1)
        self.z1 = float(z1)
        self.cell = cell
        self.nx = max(1, int((x1 - x0) / cell))
        self.nz = max(1, int((z1 - z0) / cell))
        self.blocked = [[False] * self.nz for _ in range(self.nx)]

    # --------------------------------------------------------- координаты
    def to_cell(self, x, z):
        i = int((x - self.x0) / self.cell)
        j = int((z - self.z0) / self.cell)
        return max(0, min(self.nx - 1, i)), max(0, min(self.nz - 1, j))

    def to_world(self, i, j):
        return (self.x0 + (i + 0.5) * self.cell,
                self.z0 + (j + 0.5) * self.cell)

    # ------------------------------------------------------------- разметка
    def _keep_zone_free(self, x, z, zones, r):
        return any(math.hypot(x - zx, z - zz) < r for (zx, zz) in zones)

    def block_circle(self, x, z, r, keep_zones=(), keep_r=0.0):
        i0, j0 = self.to_cell(x - r, z - r)
        i1, j1 = self.to_cell(x + r, z + r)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                wx, wz = self.to_world(i, j)
                if math.hypot(wx - x, wz - z) <= r:
                    if not self._keep_zone_free(wx, wz, keep_zones, keep_r):
                        self.blocked[i][j] = True

    def block_rect(self, x0, z0, x1, z1, keep_zones=(), keep_r=0.0):
        i0, j0 = self.to_cell(min(x0, x1), min(z0, z1))
        i1, j1 = self.to_cell(max(x0, x1), max(z0, z1))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                wx, wz = self.to_world(i, j)
                if not self._keep_zone_free(wx, wz, keep_zones, keep_r):
                    self.blocked[i][j] = True

    def free(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.nz and not self.blocked[i][j]

    def nearest_free(self, i, j):
        """Спиральный поиск ближайшей свободной клетки."""
        if self.free(i, j):
            return i, j
        for r in range(1, max(self.nx, self.nz)):
            for di in range(-r, r + 1):
                for dj in (-r, r):
                    if self.free(i + di, j + dj):
                        return i + di, j + dj
            for dj in range(-r + 1, r):
                for di in (-r, r):
                    if self.free(i + di, j + dj):
                        return i + di, j + dj
        return i, j

    # ------------------------------------------------------------- видимость
    def line_free(self, ax, az, bx, bz, margin=1.0):
        """Свободна ли прямая (с запасом margin по сторонам)."""
        dist = math.hypot(bx - ax, bz - az)
        steps = max(2, int(dist / 0.4))
        mi = int(margin / self.cell)
        for s in range(steps + 1):
            t = s / steps
            i, j = self.to_cell(ax + (bx - ax) * t, az + (bz - az) * t)
            for di in range(-mi, mi + 1):
                for dj in range(-mi, mi + 1):
                    if not self.free(i + di, j + dj):
                        return False
        return True

    # ------------------------------------------------------------------- A*
    def find_path(self, p0, p1, margin=1.0):
        """Путь (x, z) от p0 к p1: A* + сглаживание. При неудаче — повтор без отступа."""
        path = self._astar(p0, p1, margin)
        if len(path) > 1:
            return path
        return self._astar(p0, p1, 0.0)

    def _astar(self, p0, p1, margin):
        si, sj = self.nearest_free(*self.to_cell(*p0))
        gi, gj = self.nearest_free(*self.to_cell(*p1))
        nx, nz = self.nx, self.nz
        mi = int(margin / self.cell)
        blocked = self.blocked

        def passable_margin(i, j):
            for di in range(-mi, mi + 1):
                for dj in range(-mi, mi + 1):
                    if not (0 <= i + di < nx and 0 <= j + dj < nz) or \
                            blocked[i + di][j + dj]:
                        return False
            return True

        open_heap = [(0.0, si, sj)]
        g = {(si, sj): 0.0}
        came = {}
        closed = set()
        found = False
        while open_heap:
            f, i, j = heapq.heappop(open_heap)
            if (i, j) in closed:
                continue
            closed.add((i, j))
            if (i, j) == (gi, gj):
                found = True
                break
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < nx and 0 <= nj < nz) or blocked[ni][nj]:
                        continue
                    if di != 0 and dj != 0:
                        # диагональ только без «срезания» углов
                        if blocked[i + di][j] or blocked[i][j + dj]:
                            continue
                        if mi > 0 and not (passable_margin(i + di, j) and
                                           passable_margin(i, j + dj)):
                            continue
                        step = 1.41421356
                    else:
                        if mi > 0 and not passable_margin(ni, nj):
                            continue
                        step = 1.0
                    ng = g[(i, j)] + step
                    if ng < g.get((ni, nj), 1e18):
                        g[(ni, nj)] = ng
                        came[(ni, nj)] = (i, j)
                        h = math.hypot(ni - gi, nj - gj)
                        heapq.heappush(open_heap, (ng + h, ni, nj))
        if not found:
            return [tuple(p1)]

        # восстановление пути
        pts = []
        cur = (gi, gj)
        while cur != (si, sj):
            pts.append(cur)
            cur = came[cur]
        pts.append((si, sj))
        pts.reverse()
        world = [self.to_world(i, j) for (i, j) in pts]
        world[-1] = (float(p1[0]), float(p1[1]))
        return self._smooth(world)

    def _smooth(self, pts):
        """Жадное сглаживание: дальняя видимая точка вместо ломаной."""
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i][0], pts[i][1],
                                                   pts[j][0], pts[j][1]):
                j -= 1
            out.append(pts[j])
            i = j
        return out
