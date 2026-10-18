# -*- coding: utf-8 -*-
"""Панели интерфейса «Трамвайчик 3D»: окна моделей YOLO, статистика, мини-карта,
миссия, веб-камера, справка. Каждое окно включается/выключается своей клавишей."""
import math
import pygame
import pygame.gfxdraw as gfx

import config
from perception.sim_perception import CLASSES_RU, CLASS_COLOR

PAD = config.PAL
TEXT = PAD["text"]
DIM = PAD["text_dim"]

# кэш статичных текстов (динамические значения рендерятся напрямую)
_TEXT_CACHE = {}


def txt(s, app, text, size, color, pos, bold=False):
    key = (text, size, color, bold)
    surf = _TEXT_CACHE.get(key)
    if surf is None:
        if len(_TEXT_CACHE) > 600:
            _TEXT_CACHE.clear()
        surf = app.font(size, bold).render(text, True, color)
        _TEXT_CACHE[key] = surf
    s.blit(surf, pos)


# --------------------------------------------------------------------- утилиты
def fit_image(src_size, box):
    """Вписать (w, h) в прямоугольник (x, y, bw, bh) с сохранением пропорций."""
    x, y, bw, bh = box
    sw, sh = src_size
    k = min(bw / sw, bh / sh)
    iw, ih = int(sw * k), int(sh * k)
    return x + (bw - iw) // 2, y + (bh - ih) // 2, iw, ih, k


def bar(s, rect, frac, col, bg=(40, 46, 60)):
    x, y, w, h = rect
    pygame.draw.rect(s, bg, (x, y, w, h), border_radius=3)
    if frac > 0:
        pygame.draw.rect(s, col, (x, y, max(4, int(w * frac)), h), border_radius=3)


# --------------------------------------------------------------------- панели
def _stat_rows(app):
    """Строки статистики (label, value) — компактные, одна колонка"""
    r = app.renderer.stats
    st = app.tram
    wd = app.world
    npc = getattr(wd, "npc", None)
    npc_txt = (f"{npc.speed:.1f} м/с · {'ждёт' if npc.speed < 0.05 else 'в рейсе'}"
               if npc is not None else "—")
    view_txt = {"orbit": "ОРБИТА", "chase": "СЛЕДЯЩАЯ",
                "cab": "1-е ЛИЦО"}.get(getattr(app, "view_mode", "orbit"), "")
    return [
        ("FPS", f"{app.fps:.0f} · мин {app.fps_min:.0f} · {app.frame_ms:.1f} мс"),
        ("Геометрия", f"гр {r['faces']} · лин {r['lines']} · ренд {r['ms']:.1f}"),
        ("Режим", ("АВТО" if st.auto else "РУЧНОЙ") +
                  f" · время ×{getattr(app, 'time_scale', 1)} · {view_txt}"),
        ("Счёт", f"{wd.score:+d} · рейсов {st.cycles} · бигб {st.bigboxes}"),
        ("Штрафы", f"наезд {wd.hits} · стены {wd.wall_hits} · груз "
                    f"{wd.cargo_wall_hits + wd.cargo_ped_hits} · роб {wd.npc_hits}"),
        ("АКБ", f"{st.battery:.0f}% · перерывов {wd.breaks_count}"),
        ("Робот II", npc_txt),
        ("Позиция", f"x {st.pos[0]:.0f} z {st.pos[1]:.0f} · {st.speed:.1f} м/с"),
        ("Состав", f"{st.length_total():.1f} м · {st.mass():.0f} кг · Р-Т-Т-Т"),
        ("Маршрут", f"{config.ROUTE_LEN:.0f} м · {config.ROUTE_TURNS} поворота"),
        ("Склады", f"2 × {config.WAREHOUSE_W:.0f}×{config.WAREHOUSE_D:.0f}"
                    f" · H {config.WALL_H:.1f} м"),
        ("Сенсоры", f"{app.lidar_top.rays}+{app.lidar_bot.rays} луч · 2 кам · "
                     f"{wd.time:.0f} с"),
    ]


def draw_stats(s, rect, app):
    """Статистика — ОДНА компактная колонка: строки сливаются,
    шрифт значений подбирается под ширину окна."""
    x, y, w, h = rect
    rows = _stat_rows(app)
    avail = max(70, w - 80)
    vf = 11
    while vf > 8:
        fnt = app.font(vf, bold=True)
        if max(fnt.size(v)[0] for _, v in rows if v) <= avail:
            break
        vf -= 1
    fval = app.font(vf, bold=True)
    row_h = min(18, max(12, (h - 8) / len(rows)))
    yy = y + 4
    for k, v in rows:
        s.blit(app.font(11).render(k, True, DIM), (x, yy))
        if v:
            s.blit(fval.render(v, True, TEXT), (x + 80, yy + (1 if vf < 11 else 0)))
        yy += row_h


def draw_minimap(s, rect, app):
    x, y, w, h = rect
    world = app.world
    # ВЕСЬ мир — два склада + маршрут 150 м; масштаб по осям,
    # карта центрируется в доступной области панели (легенда снизу)
    wx0, wz0, wx1, wz1 = world.x0, world.z0, world.x1, world.z1
    sc = min((w - 14) / (wx1 - wx0), (h - 40) / (wz1 - wz0))
    ox = x + 7 - wx0 * sc
    map_h_px = (wz1 - wz0) * sc
    oy = y + 6 + max(0, (h - 40 - map_h_px) / 2) - wz0 * sc

    def mx(px):
        return ox + px * sc

    def my(pz):
        return oy + pz * sc

    pygame.draw.rect(s, (16, 20, 28), (x, y, w, h), border_radius=4)
    # бетон территории + зелёные полы складов
    for i in range(len(config.ROUTE_WPTS) - 1):
        (ax, az), (bx, bz) = config.ROUTE_WPTS[i], config.ROUTE_WPTS[i + 1]
        d = config.ROUTE_HALF_W
        pygame.draw.rect(s, (46, 50, 60),
                         (min(mx(ax), mx(bx)) - d * sc, min(my(az), my(bz)) - d * sc,
                          abs(bx - ax + 2 * d) * sc + (0 if abs(bz - az) < 1e-6 else 0),
                          abs(bz - az + 2 * d) * sc + (0 if abs(bx - ax) < 1e-6 else 0)))
    for cx in (config.WH_A_CX, config.WH_B_CX):
        rx0, rx1 = cx - 14, cx + 14
        pygame.draw.rect(s, (30, 52, 30),
                         (mx(rx0), my(-6.5), 28 * sc, 13 * sc), border_radius=2)
    # стеллажи / рем. столы
    for rk in world.racks:
        rx, rz = rk["pos"]
        sx, sz = rk["size"]
        pygame.draw.rect(s, (74, 86, 116), (mx(rx - sx / 2), my(rz - sz / 2),
                                            max(2, sx * sc), max(2, sz * sc)))
    for tb in getattr(world, "tables", []):
        rx, rz = tb["pos"]
        sx, sz = tb["size"]
        pygame.draw.rect(s, (120, 124, 140), (mx(rx - sx / 2), my(rz - sz / 2),
                                              max(2, sx * sc), max(2, sz * sc)))
    # маршрут (оранжевая линия по центру коридора)
    pts = [(mx(p[0]), my(p[1])) for p in config.ROUTE_WPTS]
    if len(pts) >= 2:
        pygame.draw.lines(s, (150, 120, 60), False, pts, 2)
    # малые круги разворота
    for (lcx, lcz) in (config.LOOP_A_C, config.LOOP_B_C):
        pygame.draw.circle(s, (150, 120, 60),
                           (int(mx(lcx)), int(my(lcz))),
                           max(3, int(config.LOOP_R * sc)), 1)
    # площадки A/B
    zw, zd = config.ZONE_SIZE
    for name, (zx, zz), col in (("A", world.zones["A"], PAD["pad_a"]),
                                ("B", world.zones["B"], PAD["pad_b"])):
        pygame.draw.rect(s, col, (mx(zx - zw / 2), my(zz - zd / 2),
                                  zw * sc, zd * sc), 1)
        s.blit(app.font(11, bold=True).render(name, True, col),
               (mx(zx) - 4, my(zz) - 6))
    # остаток пути автопилота
    if app.tram.path and app.tram.auto:
        pp = [(mx(p[0]), my(p[1])) for p in app.tram.path[app.tram.pi:]]
        if len(pp) >= 2:
            pygame.draw.lines(s, PAD["ok"], False, pp, 1)
    # работники / тара / штабелёры
    for ped in world.pedestrians:
        pygame.draw.circle(s, (250, 170, 70),
                           (int(mx(ped.pos[0])), int(my(ped.pos[1]))), 2)
    for cr in world.crates:
        px, pz = cr["pos"]
        col = (120, 170, 255) if cr.get("kind") == "cart" else (
            (140, 120, 100) if cr.get("kind") == "trash" else (150, 120, 80))
        sz = max(2, int(cr["size"][0] * sc))
        pygame.draw.rect(s, col, (mx(px) - sz // 2, my(pz) - sz // 2, sz, sz))
    for st in getattr(world, "stackers", []):
        pygame.draw.rect(s, (240, 150, 60), (mx(st.pos[0]) - 2, my(st.pos[1]) - 2,
                                             4, 4))
    # робот II
    npc = getattr(world, "npc", None)
    if npc is not None:
        for wp in npc.wagon_pos:
            pygame.draw.circle(s, (150, 140, 220),
                               (int(mx(wp[0])), int(my(wp[1]))), 3)
        nf = npc.fwd()
        ntip = (mx(npc.pos[0] + nf[0] * 1.2), my(npc.pos[1] + nf[1] * 1.2))
        nl = (mx(npc.pos[0] - nf[0] * 0.7 + nf[1] * 0.6),
              my(npc.pos[1] - nf[1] * 0.7 - nf[0] * 0.6))
        nr = (mx(npc.pos[0] - nf[0] * 0.7 - nf[1] * 0.6),
              my(npc.pos[1] - nf[1] * 0.7 + nf[0] * 0.6))
        pygame.draw.polygon(s, (150, 140, 220), [ntip, nl, nr])
    # состав игрока (вагоны → тягач сверху)
    t = app.tram
    for i, wp in enumerate(t.wagon_pos):
        col = PAD["wagon_body"][i % len(PAD["wagon_body"])]
        pygame.draw.circle(s, col, (int(mx(wp[0])), int(my(wp[1]))), 3)
        if t.cargo and t.cargo_count > i:
            pygame.draw.circle(s, PAD["cargo"],
                               (int(mx(wp[0])), int(my(wp[1]))), 1)
    f = t.fwd()
    tip = (mx(t.pos[0] + f[0] * 1.1), my(t.pos[1] + f[1] * 1.1))
    left = (mx(t.pos[0] - f[0] * 0.7 + f[1] * 0.55),
            my(t.pos[1] - f[1] * 0.7 - f[0] * 0.55))
    right = (mx(t.pos[0] - f[0] * 0.7 - f[1] * 0.55),
             my(t.pos[1] - f[1] * 0.7 + f[0] * 0.55))
    pygame.draw.polygon(s, PAD["tram_body"], [tip, left, right])
    # легенда (внизу панели)
    ly = y + h - 26
    s.blit(app.font(10).render(
        f"2 склада {config.WAREHOUSE_W:.0f}×{config.WAREHOUSE_D:.0f} м · "
        f"маршрут {config.ROUTE_LEN:.0f} м", True, DIM), (x + 4, ly))
    s.blit(app.font(10).render(
        "зелёное — склады · серое — территория · линии — маршрут",
        True, (110, 120, 140)), (x + 4, ly + 13))


def draw_mission(s, rect, app):
    """Задание рейса A→B — АДАПТИВНО: операции по наблюдениям (v1.5.0)."""
    x, y, w, h = rect
    t = app.tram
    wd = app.world
    state_col = {"ПОГРУЗКА": PAD["pad_a"], "СТОП-ЛИНИЯ": (255, 200, 80),
                 "ДВИЖЕНИЕ К B": PAD["accent"], "РАЗГРУЗКА": PAD["pad_b"],
                 "ВОЗВРАЩЕНИЕ К A": PAD["accent"], "IDLE": DIM}.get(t.state, TEXT)
    s.blit(app.font(14, bold=True).render(f"Статус: {t.state}", True, state_col),
           (x, y + 2))
    # прогресс операции + тайминг из наблюдений
    if t.state in ("ПОГРУЗКА", "РАЗГРУЗКА"):
        real = (config.LOAD_TIME_REAL if t.state == "ПОГРУЗКА"
                else "40 с (набл.)")
        s.blit(app.font(10).render(f"осталось {max(0.0, t.timer):.0f} с · набл.: {real}",
                                   True, DIM), (x, y + 22))
        bar(s, (x, y + 37, w - 6, 11), t.progress, state_col)
        s.blit(app.font(10).render(
            f"грузов на вагонах: {t.cargo_count}/{t.n_wagons}", True, TEXT),
            (x, y + 51))
    elif t.state == "СТОП-ЛИНИЯ":
        s.blit(app.font(10).render(
            f"остановка перед зоной: {max(0.0, t.timer):.0f} с (набл. 18–28 с)",
            True, (255, 200, 80)), (x, y + 22))
        bar(s, (x, y + 37, w - 6, 11), t.progress, (255, 200, 80))
        s.blit(app.font(10).render("звуковой сигнал подан работникам", True, DIM),
               (x, y + 51))
    else:
        path_left = max(0, len(t.path) - t.pi) if t.path else 0
        s.blit(app.font(10).render(
            f"маршрут {config.ROUTE_LEN:.0f} м · {path_left} сегм. · "
            f"{'АВТО' if t.auto else 'РУЧНОЙ (TAB)'}",
            True, DIM), (x, y + 24))
        s.blit(app.font(10).render(
            f"скорость {t.speed:.1f} из {config.TRAM_SPEED:.1f} м/с · "
            f"{'склад B' if t.state == 'ДВИЖЕНИЕ К B' else 'склад A'}",
            True, DIM), (x, y + 38))
    # батарея (наблюдение: АКБ хватает на 1,5 смены)
    bcol = PAD["ok"] if t.battery > 40 else (255, 200, 80)
    s.blit(app.font(10).render("АКБ", True, DIM), (x, y + 65))
    bar(s, (x + 30, y + 66, w - 40, 9), t.battery / 100.0, bcol)
    s.blit(app.font(10).render(f"{t.battery:.0f}%", True, bcol), (x + w - 34, y + 65))
    # рейсы и бигбоксы (наблюдение: рейс = 3 бигбокса у ворот)
    s.blit(app.font(11).render(
        f"Рейсов: {t.cycles} · бигбоксов: {t.bigboxes} "
        f"(по {config.BIGBOX_PER_TRIP}/рейс)", True, TEXT), (x, y + 80))
    # перерыв (наблюдение: цикл +5 мин)
    if wd.break_active:
        s.blit(app.font(11, bold=True).render(
            f"ПЕРЕРЫВ: {max(0.0, wd.break_left):.0f} с — цикл +5 мин",
            True, (255, 200, 80)), (x, y + 96))
    else:
        s.blit(app.font(10).render(
            f"перерывов: {wd.breaks_count} · потери цикла: {wd.break_loss} с",
            True, DIM), (x, y + 97))
    # счёт и штрафные очки
    score = wd.score
    score_col = PAD["ok"] if score >= 0 else PAD["warn"]
    s.blit(app.font(14, bold=True).render(f"Счёт: {score:+d}", True, score_col),
           (x, y + 112))
    if h >= 168:
        st_a = wd.stacker_a
        st_b = wd.stacker_b
        s.blit(app.font(10).render(
            f"штабелёры: A {st_a.state} · B {st_b.state} · убрано "
            f"{st_a._done + st_b._done}", True, DIM), (x, y + 131))
    if h >= 182:
        s.blit(app.font(10).render(
            f"наезды {wd.hits}×−{config.HIT_PENALTY} · стены "
            f"{wd.wall_hits}×−{config.WALL_HIT_PENALTY} · роб.II "
            f"{wd.npc_hits}×−{config.NPC_HIT_PENALTY}", True, DIM), (x, y + 143))
    # предупреждение (в самом низу окна)
    wy = y + h - 27
    if t.warn:
        pygame.draw.rect(s, PAD["warn"], (x, wy, w - 4, 24), width=2, border_radius=4)
        s.blit(app.font(12, bold=True).render("ПЕШЕХОД НА ПУТИ — СТОП", True, PAD["warn"]),
               (x + 8, wy + 5))
    else:
        s.blit(app.font(10).render(
            "безопасность: торможение перед людьми и роботом II", True, DIM),
            (x, wy + 7))


def draw_cam(s, rect, app):
    x, y, w, h = rect
    if app.onboard.surface is None:
        app.onboard.update(app)
    surf = app.onboard.surface
    ix, iy, iw, ih, k = fit_image(surf.get_size(), (x, y, w, h - 18))
    s.blit(app.onboard.scaled((iw, ih)), (ix, iy))
    pygame.draw.rect(s, (90, 110, 150), (ix, iy, iw, ih), 1)
    foot = f"FOV {config.ONBOARD_FOV:.0f}°  |  КАМЕРА 1 · передняя"
    ffoot = app.font(11)
    if ffoot.size(foot)[0] > w - 8:
        foot = "КАМ 1 · передняя"
    s.blit(ffoot.render(foot, True, DIM), (x, y + h - 14))


def draw_cam2(s, rect, app):
    """КАМЕРА 2: задняя, на последнем прицепе"""
    x, y, w, h = rect
    if app.rear.surface is None:
        app.rear.update(app)
    surf = app.rear.surface
    ix, iy, iw, ih, k = fit_image(surf.get_size(), (x, y, w, h - 18))
    s.blit(app.rear.scaled((iw, ih)), (ix, iy))
    pygame.draw.rect(s, (90, 110, 150), (ix, iy, iw, ih), 1)
    blink = (app.frame_n // 20) % 2 == 0
    if blink:
        pygame.draw.circle(s, PAD["warn"], (ix + iw - 10, iy + 10), 4)
    foot = f"FOV {config.CAM2_FOV:.0f}°  |  КАМЕРА 2 · задняя (прицеп 3)"
    ffoot = app.font(11)
    if ffoot.size(foot)[0] > w - 8:
        foot = "КАМ 2 · задняя · прицеп 3"
    s.blit(ffoot.render(foot, True, DIM), (x, y + h - 14))


# --------------------------------------------------------- лидары (v1.2.0)
def draw_lidar_top(s, rect, app):
    """Радар верхнего лидара: кольца, отражения по типам, развёртка.
    АДАПТИВНО: радиус и центр круга подбираются под размер окна."""
    x, y, w, h = rect
    lid = app.lidar_top
    # правая зона легенды фиксированной ширины, радар — по остатку слева
    leg_w = 78
    R = max(26, min(56, (h - 72) // 2, (w - leg_w - 24) // 2))
    cxp = x + (w - leg_w - 8) // 2
    cyp = y + 34 + R
    pygame.draw.circle(s, (20, 26, 36), (cxp, cyp), R)
    for kk in (1.0, 0.66, 0.33):
        pygame.draw.circle(s, (44, 54, 74), (cxp, cyp), max(4, int(R * kk)), 1)
    pygame.draw.line(s, (44, 54, 74), (cxp - R, cyp), (cxp + R, cyp))
    pygame.draw.line(s, (44, 54, 74), (cxp, cyp - R), (cxp, cyp + R))
    sc = R / lid.range
    for (ang, dist, kind) in lid.scan:
        if kind == "none":
            continue
        col = config.LIDAR_KIND_COLOR[kind]
        px = cxp + math.sin(ang) * dist * sc
        py = cyp - math.cos(ang) * dist * sc
        pygame.draw.circle(s, col, (int(px), int(py)), 2)
    a = lid.angle
    pygame.draw.line(s, config.LIDAR_TOP_COLOR, (cxp, cyp),
                     (cxp + math.sin(a) * R, cyp - math.cos(a) * R), 2)
    f = app.tram.fwd()
    pygame.draw.line(s, PAD["tram_body"], (cxp, cyp),
                     (cxp + f[0] * 12, cyp - f[1] * 12), 2)
    pygame.draw.circle(s, PAD["tram_body"], (cxp, cyp), 3)
    lab = f"{lid.label} · {lid.rays} лучей · {lid.range:.0f} м"
    flab = app.font(11)
    if flab.size(lab)[0] > w - 8:
        lab = f"{lid.label} · {lid.rays} л · {lid.range:.0f} м"
    txt(s, app, lab, 11, config.LIDAR_TOP_COLOR, (x + 4, y))
    nd, nk = lid.nearest()
    col = config.LIDAR_KIND_COLOR.get(nk, DIM) if nk != "none" else DIM
    s.blit(app.font(11).render(
        f"ближайшее: {nd:4.1f} м · {config.LIDAR_KIND_RU[nk]}", True, col),
        (x + 4, y + 15))
    # легенда типов (правый столбец)
    lx, ly = x + w - leg_w + 4, y + 20
    for kind in ("person", "crate", "rack", "wall"):
        pygame.draw.circle(s, config.LIDAR_KIND_COLOR[kind], (lx, ly + 5), 3)
        s.blit(app.font(9).render(config.LIDAR_KIND_RU[kind], True, DIM),
               (lx + 8, ly))
        ly += 13
    nh = len([1 for (_a, _d, k2) in lid.scan if k2 != "none"])
    s.blit(app.font(9).render(f"отражений: {nh}", True, DIM), (lx - 4, ly + 4))


def draw_lidar_bottom(s, rect, app):
    """Нижний лидар: веер лучей по полу + профиль дальности.
    АДАПТИВНО: радиус веера подбирается под высоту окна."""
    x, y, w, h = rect
    lid = app.lidar_bot
    lab = f"{lid.label} · {lid.rays} лучей · {lid.range:.0f} м"
    flab = app.font(11)
    if flab.size(lab)[0] > w - 8:
        lab = f"{lid.label} · {lid.rays} л"
    txt(s, app, lab, 11, config.LIDAR_BOTTOM_COLOR, (x + 4, y))
    nd, nk = lid.nearest()
    col = config.LIDAR_KIND_COLOR.get(nk, DIM) if nk != "none" else DIM
    near = f"мин {nd:4.1f} м · {config.LIDAR_KIND_RU[nk]}"
    fnear = app.font(9)
    if flab.size(lab)[0] + fnear.size(near)[0] + 18 > w:
        near = f"мин {nd:4.1f} м"          # короткий вариант на узком окне
    if flab.size(lab)[0] + fnear.size(near)[0] + 18 <= w:
        s.blit(fnear.render(near, True, col),
               (x + w - fnear.size(near)[0] - 6, y + 3))
    R = max(24, min(62, (h - 66) / 2 - 2, (w - 24) / 2))
    ox, oy = x + w // 2, y + 30 + R
    sc = R / lid.range
    for (ang, dist, kind) in lid.scan:
        ex = ox + math.sin(ang) * dist * sc
        ey = oy - math.cos(ang) * dist * sc
        c2 = config.LIDAR_BOTTOM_COLOR if kind != "none" else (52, 74, 52)
        pygame.draw.line(s, c2, (ox, oy), (ex, ey), 1)
        if kind != "none":
            pygame.draw.circle(s, config.LIDAR_KIND_COLOR[kind],
                               (int(ex), int(ey)), 2)
    # границы сектора
    half = lid.fov * 0.5
    base = lid.scan[0][0] if lid.scan else 0.0
    for edge in (base - half, base + half):
        pygame.draw.line(s, (70, 88, 128), (ox, oy),
                         (ox + math.sin(edge) * R, oy - math.cos(edge) * R), 1)
    # профиль дальности (полоса внизу)
    lbl_y = y + h - 25
    txt(s, app, "профиль дальности по сектору:", 9, DIM, (x + 4, lbl_y))
    py0 = y + h - 7
    ph = 16
    n = len(lid.scan)
    if n:
        pts = []
        for i, (ang, dist, kind) in enumerate(lid.scan):
            px = x + 8 + (w - 16) * i / max(1, n - 1)
            py = py0 - (dist / lid.range) * ph
            pts.append((px, py))
        pygame.draw.lines(s, (90, 210, 255), False, pts, 1)
    pygame.draw.line(s, (44, 54, 74), (x + 6, py0), (x + w - 6, py0), 1)


# --------------------------------------------------------- окна моделей YOLO
_FONT_CACHE = {}


def _small_font(size):
    f = _FONT_CACHE.get(size)
    if f is None:
        f = pygame.font.Font(None, size)
        _FONT_CACHE[size] = f
    return f


def _overlay(s, task, out, ox, oy, k, app=None):
    """Детальный оверлей задачи поверх масштабированного изображения
    бортовой камеры (треки, центры, углы, индексы точек)."""
    if task == "detect":
        f = _small_font(max(14, int(13 * k)))
        # хвосты треков пешеходов (панель показывает движение)
        if app is not None:
            for tr in getattr(app.onboard, "trails", {}).values():
                if len(tr) >= 2:
                    pts = [(ox + px * k, oy + py * k) for (px, py) in tr]
                    pygame.draw.lines(s, (90, 220, 130), False, pts, 1)
        for d in out["detect"]:
            bx, by, bw, bh = d["box"]
            col = CLASS_COLOR.get(d["label"], (200, 200, 200))
            r = pygame.Rect(int(ox + bx * k), int(oy + by * k),
                            int(bw * k), int(bh * k))
            pygame.draw.rect(s, col, r, 2)
            # уголки рамки (детальный стиль)
            for (cxx, cyy, dx2, dy2) in ((r.x, r.y, 1, 1), (r.right - 1, r.y, -1, 1),
                                         (r.x, r.bottom - 1, 1, -1),
                                         (r.right - 1, r.bottom - 1, -1, -1)):
                pygame.draw.line(s, col, (cxx, cyy), (cxx + dx2 * 6, cyy), 2)
                pygame.draw.line(s, col, (cxx, cyy), (cxx, cyy + dy2 * 6), 2)
            tag = f"{CLASSES_RU.get(d['label'], d['label'])} {d['conf']:.2f}"
            if "id" in d:
                tag += f" #{d['id']}"
            tw = f.size(tag)[0]
            ty = max(0, r.y - 14)
            pygame.draw.rect(s, col, (r.x, ty, tw + 6, 13))
            s.blit(f.render(tag, True, (20, 24, 30)), (r.x + 3, ty))
            # центр объекта
            ccx, ccy = r.center
            pygame.draw.line(s, col, (ccx - 3, ccy), (ccx + 3, ccy))
            pygame.draw.line(s, col, (ccx, ccy - 3), (ccx, ccy + 3))
    elif task == "obb":
        f = _small_font(max(12, int(11 * k)))
        for d in out["obb"]:
            col = CLASS_COLOR.get(d["label"], (200, 200, 200))
            cx, cy = ox + d["cx"] * k, oy + d["cy"] * k
            hw, hh = d["w"] * k / 2, d["h"] * k / 2
            ca, sa = math.cos(d["angle"]), math.sin(d["angle"])
            pts = []
            for (px, py) in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
                pts.append((cx + px * ca - py * sa, cy + px * sa + py * ca))
            pygame.draw.polygon(s, col, pts, 2)
            pygame.draw.circle(s, col, (int(cx), int(cy)), 2)
            ang = math.degrees(d["angle"])
            s.blit(f.render(f"{ang:.0f}°", True, col), (int(cx) + 4, int(cy) - 9))
    elif task == "segment":
        for d in out["segment"]:
            col = CLASS_COLOR.get(d["label"], (200, 200, 200))
            pts = [(ox + px * k, oy + py * k) for (px, py) in d["pts"]]
            if len(pts) >= 3:
                gfx.filled_polygon(s, [(int(px), int(py)) for (px, py) in pts],
                                   (col[0], col[1], col[2], 70))
                pygame.draw.polygon(s, col, pts, 2)
    elif task == "pose":
        f = _small_font(max(11, int(10 * k)))
        for pi, d in enumerate(out["pose"]):
            pts = [(ox + p[0] * k, oy + p[1] * k, p[2]) for p in d["pts"]]
            for (a, b) in d["links"]:
                if pts[a][2] and pts[b][2]:
                    pygame.draw.line(s, (90, 220, 130), (pts[a][0], pts[a][1]),
                                     (pts[b][0], pts[b][1]), 2)
            for ki, (px, py, okv) in enumerate(pts):
                if okv:
                    gfx.filled_circle(s, int(px), int(py), 3, (240, 240, 120))
            vis = [p for p in pts if p[2]]
            if vis:
                top = min(vis, key=lambda p: p[1])
                s.blit(f.render(f"P{pi + 1}", True, (240, 240, 120)),
                       (int(top[0]) + 5, int(top[1]) - 4))
    elif task == "semantic":
        for d in out["semantic"]:
            pts = [(ox + px * k, oy + py * k) for (px, py) in d["pts"]]
            if len(pts) >= 3:
                col = d["col"]
                gfx.filled_polygon(s, [(int(px), int(py)) for (px, py) in pts],
                                   (col[0], col[1], col[2], 90))
                pygame.draw.polygon(s, d["col"], pts, 1)
    elif task == "depth":
        for (px, py, size, gray) in out["depth"]:
            g = gray
            pygame.draw.rect(s, (g, g, g),
                             (ox + px * k - size * k / 2, oy + py * k - size * k / 2,
                              max(2, size * k), max(2, size * k)))


# ------------------------------------------------- детальные панели (v1.2.0)
def _agg_confs(out, key):
    """Агрегация по классам: [(label, средний conf, количество)]."""
    agg = {}
    for d in out.get(key, []):
        lbl = d["label"]
        c, n = agg.get(lbl, (0.0, 0))
        agg[lbl] = (c + d.get("conf", 1.0), n + 1)
    return [(lbl, c / n, n) for lbl, (c, n) in
            sorted(agg.items(), key=lambda kv: -kv[1][1])]


def _poly_area(pts):
    """Площадь полигона (формула шнурования) — для оценки покрытия."""
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return abs(a) * 0.5


def _task_meta(app, task):
    """(имя модели, мс задачи, Гц конвейера) — витрина метрик модели."""
    base = config.MODEL_BASE_MS
    total = sum(base.values())
    if app.yolo_ms > 0.2:
        return (config.YOLO_MODEL_NAMES[task],
                base[task] / total * app.yolo_ms, 1000.0 / app.yolo_ms)
    return config.YOLO_MODEL_NAMES[task], base[task], 1000.0 / total


def _sidebar_title(s, app, x, y, w, text):
    pygame.draw.line(s, (54, 66, 92), (x, y + 8), (x + w, y + 8))
    txt(s, app, text, 10, PAD["accent"], (x, y))


def _conf_slider(s, app, x, y, w):
    """Индикатор порога уверенности под изображением."""
    thr = 0.35
    pygame.draw.rect(s, (40, 46, 60), (x, y + 5, int(w * thr), 3))
    pygame.draw.line(s, (60, 76, 110), (x + int(w * thr), y + 6),
                     (x + w, y + 6), 1)
    pygame.draw.circle(s, (255, 210, 110), (x + int(w * thr), y + 6), 4)
    txt(s, app, f"порог conf ≥ {thr:.2f}", 9, DIM, (x, y + 11))


def _side_list(s, app, out, key, x, y, w, h, title):
    """Список классов: цветовой чип + имя + количество + средний conf."""
    rows = _agg_confs(out, key)
    _sidebar_title(s, app, x, y, w, title)
    yy = y + 16
    for lbl, conf, n in rows[:9]:
        col = CLASS_COLOR.get(lbl, (200, 200, 200))
        pygame.draw.rect(s, col, (x, yy + 2, 8, 8), border_radius=2)
        s.blit(app.font(11).render(CLASSES_RU.get(lbl, lbl), True, TEXT), (x + 13, yy))
        s.blit(app.font(11, bold=True).render(f"x{n}", True, TEXT), (x + w - 46, yy))
        s.blit(app.font(10).render(f"{conf:.2f}", True, DIM), (x + w - 24, yy + 1))
        yy += 16
    if not rows:
        s.blit(app.font(11).render("— пусто —", True, DIM), (x, yy))
        yy += 16
    return yy


def _side_detect(s, app, out, x, y, w, h):
    yy = _side_list(s, app, out, "detect", x, y, w, h, "КЛАССЫ / СЧЁТ")
    n_tr = sum(len(v) for v in getattr(app.onboard, "trails", {}).values())
    _sidebar_title(s, app, x, yy + 6, w, "ТРЕКИ")
    s.blit(app.font(11).render(f"хвостов: {n_tr}", True, TEXT), (x, yy + 22))
    s.blit(app.font(10).render("зелёные линии на кадре", True, DIM), (x, yy + 36))


def _side_pose(s, app, out, x, y, w, h):
    _sidebar_title(s, app, x, y, w, "ЛЮДИ / ТОЧКИ")
    yy = y + 16
    for i, d in enumerate(out.get("pose", [])[:10]):
        vis = sum(1 for p in d["pts"] if p[2])
        s.blit(app.font(11).render(f"P{i + 1}", True, (240, 240, 120)), (x, yy))
        bar(s, (x + 26, yy + 3, w - 62, 8), vis / 15.0, PAD["ok"])
        s.blit(app.font(10).render(f"{vis}/15", True, DIM), (x + w - 32, yy))
        yy += 16
    if not out.get("pose"):
        s.blit(app.font(11).render("— пусто —", True, DIM), (x, yy))
        yy += 16
    s.blit(app.font(10).render("связей: 14 на скелет", True, DIM), (x, yy + 4))


def _side_segment(s, app, out, x, y, w, h):
    yy = _side_list(s, app, out, "segment", x, y, w, h, "МАСКИ / КЛАССЫ")
    area = sum(_poly_area(d["pts"]) for d in out.get("segment", []))
    cov = min(1.0, area / (480.0 * 320.0))
    _sidebar_title(s, app, x, yy + 6, w, "ПОКРЫТИЕ КАДРА")
    bar(s, (x, yy + 24, w - 8, 10), cov, PAD["accent"])
    s.blit(app.font(11).render(f"{cov * 100:.1f}%", True, TEXT), (x, yy + 38))


def _side_classify(s, app, out, x, y, w, h):
    _sidebar_title(s, app, x, y, w, "TOP-5 КЛАССОВ")
    rank = out.get("classify", [])
    yy = y + 16
    for lbl, p in rank[:5]:
        s.blit(app.font(11).render(str(lbl), True, TEXT), (x, yy))
        bar(s, (x, yy + 14, w - 36, 8), p, PAD["accent"])
        s.blit(app.font(10).render(f"{p * 100:.0f}%", True, DIM), (x + w - 30, yy + 12))
        yy += 32
    if rank:
        ent = -sum(p * math.log(p + 1e-9) for _, p in rank if p > 0)
        _sidebar_title(s, app, x, yy + 2, w, "ЭНТРОПИЯ")
        s.blit(app.font(11).render(f"{ent:.2f} бит", True, TEXT), (x, yy + 18))


def _side_obb(s, app, out, x, y, w, h):
    _sidebar_title(s, app, x, y, w, "ОРИЕНТАЦИЯ / УГОЛ")
    yy = y + 16
    for d in out.get("obb", [])[:11]:
        col = CLASS_COLOR.get(d["label"], (200, 200, 200))
        pygame.draw.rect(s, col, (x, yy + 2, 8, 8), border_radius=2)
        s.blit(app.font(11).render(CLASSES_RU.get(d["label"], d["label"]),
                                   True, TEXT), (x + 13, yy))
        ang = math.degrees(d["angle"])
        s.blit(app.font(11).render(f"{ang:+5.0f}°", True, (255, 210, 110)),
               (x + w - 44, yy))
        yy += 15
    if not out.get("obb"):
        s.blit(app.font(11).render("— пусто —", True, DIM), (x, yy))


def _side_semantic(s, app, out, x, y, w, h):
    _sidebar_title(s, app, x, y, w, "ЛЕГЕНДА / ПОКРЫТИЕ")
    yy = y + 16
    for d in out.get("semantic", [])[:10]:
        pygame.draw.rect(s, d["col"], (x, yy + 2, 8, 8), border_radius=2)
        cov = min(1.0, _poly_area(d["pts"]) / (480.0 * 320.0))
        s.blit(app.font(11).render(str(d["label"]), True, TEXT), (x + 13, yy))
        s.blit(app.font(10).render(f"{cov * 100:.0f}%", True, DIM), (x + w - 28, yy + 1))
        yy += 16
    if not out.get("semantic"):
        s.blit(app.font(11).render("— пусто —", True, DIM), (x, yy))


def _side_depth(s, app, out, x, y, w, h):
    _sidebar_title(s, app, x, y, w, "ШКАЛА ГЛУБИНЫ")
    bx, by0 = x + 4, y + 18
    by1 = y + h - 44
    for yy in range(by0, max(by0 + 1, by1)):
        t_ = (yy - by0) / max(1, by1 - by0 - 1)     # верх = близко
        g = int(255 - t_ * 233)
        pygame.draw.line(s, (g, g, g), (bx, yy), (bx + 16, yy))
    s.blit(app.font(10).render("0.5 м", True, DIM), (bx + 22, by0 - 2))
    s.blit(app.font(10).render("26 м", True, DIM), (bx + 22, by1 - 8))
    pts = out.get("depth", [])
    if pts:
        ccx, ccy = 240, 160
        px_, py_, _sz, gray = min(pts, key=lambda p: (p[0] - ccx) ** 2 + (p[1] - ccy) ** 2)
        d_c = (255 - gray) / 9.2
        ds = [(255 - g) / 9.2 for (*_p, g) in pts]
        _sidebar_title(s, app, x, y + h - 40, w, "ДИСТАНЦИИ")
        s.blit(app.font(11).render(f"центр: {d_c:4.1f} м", True, TEXT), (x, y + h - 24))
        s.blit(app.font(10).render(
            f"мин {min(ds):.1f} · ср {sum(ds) / len(ds):.1f} м", True, DIM),
            (x, y + h - 11))


SIDES = {"detect": _side_detect, "pose": _side_pose, "segment": _side_segment,
         "classify": _side_classify, "obb": _side_obb, "semantic": _side_semantic,
         "depth": _side_depth}


def model_panel(task):
    """Фабрика ДЕТАЛЬНЫХ панелей задач YOLO (клавиши 1..7):
    кадр камеры + оверлей + боковая статистика + метрики модели.
    АДАПТИВНО: ширина боковой панели зависит от ширины окна."""
    def draw(s, rect, app):
        x, y, w, h = rect
        out = app.onboard.outputs
        if out is None:                  # первый кадр — считаем восприятие
            app.onboard.update(app)
            out = app.onboard.outputs
        if not out:
            out = {t: [] for t in config.TASKS}
        model, ms, hz = _task_meta(app, task)
        # шапка: название слева, метрики справа (шрифт мельче на узких окнах)
        txt(s, app, f"YOLO · {config.TASKS_RU[task]}", 12, PAD["accent"],
            (x, y), bold=True)
        meta = f"{model} · {ms:.1f} мс · {hz:.0f} Гц"
        mf = app.font(10) if w >= 360 else app.font(8)
        s.blit(mf.render(meta, True, DIM),
               (x + w - mf.size(meta)[0] - 2, y + (1 if w >= 360 else 3)))
        # боковая панель статистики: 136px на широких окнах, ~1/3 на узких
        sb = 136 if w >= 400 else max(92, w // 3)
        img_box = (x, y + 18, w - sb - 6, h - 18)
        if app.onboard.surface is None:
            app.onboard.update(app)
        surf = app.onboard.surface
        ix, iy, iw, ih, k = fit_image(surf.get_size(), img_box)
        s.blit(app.onboard.scaled((iw, ih)), (ix, iy))
        pygame.draw.rect(s, (90, 110, 150), (ix, iy, iw, ih), 1)
        _overlay(s, task, out, ix, iy, k, app)
        # подпись под изображением
        by = iy + ih + 5
        note = {"detect": "треки · центры · порог", "pose": "15 точек, COCO-подобные",
                "segment": "выпуклые маски классов", "classify": "сцена целиком",
                "obb": "угол поворота рамок", "semantic": "зоны A/B + классы",
                "depth": "темнее = дальше"}.get(task, "")
        if task in ("detect", "obb", "segment"):
            _conf_slider(s, app, ix + 2, by, iw - 90)
            s.blit(app.font(9).render(note, True, DIM), (ix + iw - 86, by + 11))
        else:
            s.blit(app.font(9).render(note, True, DIM), (ix + 2, by + 4))
        SIDES[task](s, app, out, x + w - sb - 4, y + 18, sb - 4, h - 18)
    return draw


def draw_webcam(s, rect, app):
    x, y, w, h = rect
    src, yolo = app.webcam, app.real_yolo
    if src is None or not src.ok:
        msg = (src.err if src and src.err else "нажмите V — подключить веб-камеру")
        s.blit(app.font(13).render(f"Веб-камера недоступна: {msg}", True, DIM),
               (x + 8, y + 8))
        s.blit(app.font(11).render(
            "режим имитации: бортовая камера слева продолжает работать", True, DIM),
            (x + 8, y + 30))
        return
    frame = src.read()
    if frame is None:
        s.blit(app.font(12).render("кадры не поступают...", True, DIM), (x + 8, y + 8))
        return
    if yolo is not None and app.frame_n % config.INFER_EVERY == 0:
        yolo.ensure_loaded()
        if yolo.available:
            ann, app.cam_labels = yolo.infer(frame)
            if ann is not None:
                frame = ann
    rgb = src.cv2.cvtColor(frame, src.cv2.COLOR_BGR2RGB)
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    ix, iy, iw, ih, k = fit_image(surf.get_size(), (x, y, w, h - 34))
    s.blit(pygame.transform.scale(surf, (iw, ih)), (ix, iy))
    pygame.draw.rect(s, (90, 110, 150), (ix, iy, iw, ih), 1)
    status = ("REC" if app.recorder.active else "готово")
    col = PAD["warn"] if app.recorder.active else DIM
    if app.recorder.active:
        pygame.draw.circle(s, PAD["warn"], (ix + iw - 12, iy + 12), 5)
    s.blit(app.font(11, bold=True).render(
        f"{status} · кадров в датасете: {app.recorder.count}", True, col),
        (x, y + h - 26))
    s.blit(app.font(10).render(
        ("модель: " + (yolo.err if yolo and yolo.err else
                       ("yolov8n активна" if yolo and yolo.available else "нет ultralytics"))) +
        "  |  O — запись", True, DIM), (x, y + h - 13))


def draw_help(s, rect, app):
    x, y, w, h = rect
    cols = [
        ("Режимы и движение", [
            ("TAB", "автопилот / ручной"),
            ("W / ↑", "газ (вперёд)"),
            ("S / ↓", "ТОРМОЗ — реверс запрещён"),
            ("A / D  <-/->", "поворот (ручной)"),
            ("ПРОБЕЛ", "стоп (ручной)"),
            ("T", "ускорение времени ×1/×2/×4/×8"),
            ("F", "вид: орбита → следящая → КАБИНА"),
            ("R", "сброс состава в зону A"),
            ("+100", "рейс (3 бигбокса) в B"),
            ("−25", "наезд на работника"),
            ("−15", "кузов о стену/стеллаж"),
            ("−30", "столкновение с роботом II"),
        ]),
        ("Производство (наблюдения)", [
            ("Склады", "два × 28 × 13 м, H 4,5 м"),
            ("Маршрут", "150 м · 4 поворота · барьеры"),
            ("Малый круг", "разворот в зонах A и B"),
            ("Зоны A/B", "обычный пол, груз ставят сюда"),
            ("", "работники убирают на штабелёрах"),
            ("Операции", "разгрузка 40 с, стоп-линия"),
            ("", "погрузка ~60–70 с/паллета"),
            ("Рейс", "3 бигбокса → отправка"),
            ("Перерыв", "цикл +5 мин (учёт потерь)"),
            ("АКБ", "1,5 смены · зарядка у ворот"),
            ("Тележки", "синие — выкат при разгрузке"),
            ("Робот II", "второй состав — уступает"),
        ]),
        ("Камера 3D", [
            ("ЛКМ (тащить)", "облёт сцены (орбита)"),
            ("Колесо", "приближение"),
            ("M", "следящая / свободная (орбита)"),
            ("F", "вид от 1-го лица — из кабины"),
            ("F11", "полный экран"),
            ("", "в кабине мышь не вращает камеру"),
        ]),
        ("Окна (вкл/выкл)", [
            ("F1", "эта справка"),
            ("F2", "статистика (справа снизу)"),
            ("F3", "мини-карта (справа сверху)"),
            ("F4", "задание рейса (над статистикой)"),
            ("F6", "упорядочить окна (адаптивно)"),
            ("8 / 9", "лидар верхний / нижний (слева)"),
            ("C / X", "камера 1 передняя / 2 задняя"),
            ("1..7", "YOLO в центре: detect, pose,"),
            ("", "segment, classify, obb, semantic, depth"),
            ("V", "веб-камера ноутбука"),
            ("O", "запись датасета (YOLO-формат)"),
            ("", "train_finetune.py — дообучение"),
        ]),
    ]
    col_w = w // len(cols)
    for ci, (title, rows) in enumerate(cols):
        cx = x + ci * col_w
        txt(s, app, title, 13, PAD["accent"], (cx, y + 2), bold=True)
        yy = y + 24
        for k, v in rows:
            if k:
                txt(s, app, k, 11, (255, 210, 110), (cx, yy), bold=True)
            txt(s, app, v, 10, TEXT, (cx + 84 if k else cx + 4, yy))
            yy += 16


# --------------------------------------------------- адаптивная раскладка
# Схема:  СЛЕВА (сверху вниз): лидар верхний, лидар нижний,
# камера 1, камера 2.  СПРАВА (сверху вниз): мини-карта, задание,
# компактная статистика.  ЦЕНТР: сетка YOLO 1-7 + веб-камера, размер
# которой подстраивается под число ОТКРЫТЫХ окон (одно окно — компактная
# панель по центру, все окна — плотная сетка без пустот).
GRID_IDS = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "webcam"]


def layout_all(app, force=False):
    """Пересчитать прямоугольники всех окон под текущий размер экрана
    (app.w × app.h). Окна, перетащенные вручную (moved), не трогаются,
    если force=False. Вызывается при старте, ресайзе, переключении окон и F6."""
    W, H = app.w, app.h
    wm = app.wm
    m = 8                      # внешний отступ
    g = 8                      # промежуток между окнами
    top = 36                   # ниже верхней панели (30px)
    bot = H - 32               # выше нижней панели (26px)
    hu = max(240, bot - top)   # доступная высота

    # --- ширина боковых колонок ---
    side_w = int(max(200, min(300, round(W * 0.165))))

    # --- ЛЕВАЯ колонка: 2 лидара + 2 камеры ---
    avail = hu - 3 * g
    rects = {
        "lidar_top": (m, top, side_w, avail * 0.27),
        "lidar_bot": (m, top + avail * 0.27 + g, side_w, avail * 0.27),
        "cam": (m, top + 2 * (avail * 0.27 + g), side_w, avail * 0.23),
        "cam2": (m, top + 2 * (avail * 0.27 + g) + avail * 0.23 + g,
                 side_w, avail * 0.23),
    }

    # --- ПРАВАЯ колонка: мини-карта / задание / статистика (компактная) ---
    rx = W - m - side_w
    stats_h = max(190, min(216, 12 * 16 + 16))         # 12 строк одной колонкой
    mission_h = max(200, min(232, hu - stats_h - 150 - 2 * g))
    map_h = max(140, min(260, hu - stats_h - mission_h - 2 * g))
    yy = top
    rects["minimap"] = (rx, yy, side_w, map_h)
    yy += map_h + g
    rects["mission"] = (rx, yy, side_w, mission_h)
    yy += mission_h + g
    rects["stats"] = (rx, yy, side_w, min(stats_h, bot - yy))

    # --- ЦЕНТР: адаптивная сетка под число ОТКРЫТЫХ окон ---
    x0 = m + side_w + g
    x1 = rx - g
    cw_total = x1 - x0
    ch_total = hu
    n_vis = sum(1 for wid in GRID_IDS if wid in wm.index and wm.index[wid].visible)
    n_vis = max(1, n_vis)
    cols = 1 if n_vis == 1 else 2
    rows = (n_vis + cols - 1) // cols
    cell_w = min((cw_total - g * (cols - 1)) / cols, 700.0)
    cell_h = min((ch_total - g * (rows - 1)) / rows, 430.0)
    # блок сетки центрируем в центральной зоне
    block_w = cols * cell_w + (cols - 1) * g
    block_h = rows * cell_h + (rows - 1) * g
    bx0 = x0 + (cw_total - block_w) / 2.0
    by0 = top + (ch_total - block_h) / 2.0
    vis_ids = [wid for wid in GRID_IDS
               if wid in wm.index and wm.index[wid].visible]
    for i, wid in enumerate(GRID_IDS):
        # скрытые окна делят слоты видимых (невидимы — пересчитаются при
        # включении); видимые получают свои ячейки по порядку
        slot = (vis_ids.index(wid) if wid in vis_ids
                else GRID_IDS.index(wid) % max(1, len(vis_ids)))
        row, col = slot // cols, slot % cols
        rects[wid] = (bx0 + col * (cell_w + g), by0 + row * (cell_h + g),
                      cell_w, cell_h)

    # --- справка: по центру экрана ---
    hw = min(780, cw_total)
    hh = min(352, hu - 8)
    rects["help"] = ((W - hw) // 2, top + (hu - hh) // 2, hw, hh)

    for wid, r in rects.items():
        win = wm.index.get(wid)
        if win is None:
            continue
        if not force and getattr(win, "moved", False):
            continue            # пользователь перетащил — не трогаем
        win.apply_rect(r)


def reset_layout(app):
    """F6: вернуть ВСЕ окна в адаптивную раскладку."""
    for wv in app.wm.wins:
        wv.moved = False
    layout_all(app, force=True)


# --------------------------------------------------------------------- реестр
def register_all(app):
    """Регистрация окон; расстановка — через layout_all (адаптивно)."""
    wm = app.wm
    P = (8, 36, 260, 180)      # placeholder — сразу заменяется layout_all
    C = (8, 36, 480, 200)      # placeholder для центральных
    # --- левая колонка: лидары и камеры ---
    wm.add("lidar_top", "Лидар ВЕРХНИЙ · 360° (крыша)", P, "8",
           draw_lidar_top, True)
    wm.add("lidar_bot", "Лидар НИЖНИЙ · пол/ноги", P, "9",
           draw_lidar_bottom, True)
    wm.add("cam", "Камера 1 · передняя (бортовая)", P, "C", draw_cam, True)
    wm.add("cam2", "Камера 2 · задняя (прицеп 3)", P, "X", draw_cam2, True)
    # --- правая колонка: карта/задание/статистика обновляются реже (FPS) ---
    wm.add("minimap", "Мини-карта: 2 склада + маршрут 150 м", P, "F3",
           draw_minimap, True, redraw_every=3)
    wm.add("mission", "Задание: доставка A → B", P, "F4", draw_mission, True,
           redraw_every=2)
    wm.add("stats", "Статистика / производительность", P, "F2", draw_stats, True,
           redraw_every=3)
    # --- центр: YOLO 1-7 по порядку (по умолчанию активно только окно 1) ---
    titles = [
        "1 · YOLO detect — детальный", "2 · YOLO pose — детальный",
        "3 · YOLO segment — детальный", "4 · YOLO classify — детальный",
        "5 · YOLO obb — детальный", "6 · YOLO semantic — детальный",
        "7 · YOLO depth — детальный",
    ]
    for i, (wid, title) in enumerate(zip(GRID_IDS[:7], titles)):
        wm.add(wid, title, C, str(i + 1), model_panel(config.TASKS[i]),
               visible=(i == 0))
    wm.add("webcam", "Веб-камера ноутбука (реальный режим)", C, "V",
           draw_webcam, False)
    wm.add("help", "Справка — управление (перетаскивайте окна за заголовок)",
           C, "F1", draw_help, False)
    layout_all(app, force=True)
