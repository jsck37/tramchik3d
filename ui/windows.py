# -*- coding: utf-8 -*-
"""Оконный менеджер UI: перетаскиваемые панели, включение/выключение окон."""
import pygame


class Win:
    __slots__ = ("id", "title", "rect", "visible", "draw", "key", "surf",
                 "title_surf", "moved", "_cache", "redraw_every")

    def __init__(self, wid, title, rect, key, draw, visible, redraw_every=2):
        self.id = wid
        self.title = title
        self.rect = list(rect)   # [x, y, w, h]
        self.visible = visible
        self.draw = draw
        self.key = key
        self.surf = None
        self.title_surf = None
        self.moved = False       # True — пользователь перетащил окно вручную
        self._cache = None       # кэш содержимого (перерисовка раз в N кадров)
        self.redraw_every = max(1, redraw_every)

    def titlebar(self):
        return pygame.Rect(self.rect[0], self.rect[1], self.rect[2], 22)

    def content(self):
        return (6, 24, self.rect[2] - 12, self.rect[3] - 30)

    def apply_rect(self, rect):
        """Применить прямоугольник адаптивной раскладки [x, y, w, h]."""
        self.rect[:] = [int(v) for v in rect]
        self.surf = None          # пересоздать поверхность под новый размер
        self.title_surf = None


class WindowManager:
    TB = 22

    def __init__(self, app):
        self.app = app
        self.wins = []
        self.index = {}
        self.drag = None        # (id, offx, offy)

    def ensure_scratch(self, w, h):
        pass  # окна рисуются в собственные поверхности

    # ------------------------------------------------------------------ реестр
    def add(self, wid, title, rect, key, draw, visible=False, redraw_every=2):
        w = Win(wid, title, rect, key, draw, visible, redraw_every)
        self.wins.append(w)
        self.index[wid] = w
        return w

    def toggle(self, wid):
        w = self.index.get(wid)
        if w is None:
            return
        w.visible = not w.visible
        self.app.toast(("Окно: " if w.visible else "Окно закрыто: ") + w.title)

    def toggle_by_key(self, keyname):
        for w in self.wins:
            if w.key == keyname:
                self.toggle(w.id)
                return True
        return False

    def any_visible(self, ids):
        return any(self.index[i].visible for i in ids if i in self.index)

    # ------------------------------------------------------------------ события
    def handle_event(self, e):
        """True — событие поглощено окном (перетаскивание за заголовок)."""
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for w in reversed(self.wins):
                if w.visible and w.titlebar().collidepoint(e.pos):
                    self.drag = (w.id, e.pos[0] - w.rect[0], e.pos[1] - w.rect[1])
                    return True
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.drag:
            self.drag = None
            return True
        elif e.type == pygame.MOUSEMOTION and self.drag:
            wid, ox, oy = self.drag
            w = self.index[wid]
            sw, sh = self.app.screen.get_size()
            w.rect[0] = max(0, min(sw - 60, e.pos[0] - ox))
            w.rect[1] = max(26, min(sh - 40, e.pos[1] - oy))
            w.moved = True        # окно покидает адаптивную раскладку (F6 — вернуть)
            return True
        return False

    # ------------------------------------------------------------------ отрисовка
    def draw(self, surface):
        """Каждое окно — отдельная НЕПРОЗРАЧНАЯ поверхность (быстрый блит без
        альфа-смешивания; v1.2.0 — оптимизация под 60+ FPS при 12+ окнах).
        v1.5.0: содержимое статичных панелей перерисовывается каждый 2-й кадр
        (кэш-блит на промежуточных) — экономия ~2 мс/кадр при 8+ окнах."""
        frame = self.app.frame_n
        for w in self.wins:
            if not w.visible:
                continue
            x, y, ww, hh = w.rect
            if w.surf is None or w.surf.get_size() != (ww, hh) or \
                    w.surf.get_flags() & pygame.SRCALPHA:
                w.surf = pygame.Surface((ww, hh))
                w.title_surf = None
                w._cache = None
            s = w.surf
            cached = getattr(w, "_cache", None)
            redraw = cached is None or frame % w.redraw_every == 0
            if redraw:
                # корпус
                s.fill((15, 18, 26))
                pygame.draw.rect(s, (88, 108, 150), (0, 0, ww, hh), width=1,
                                 border_radius=8)
                # заголовок
                pygame.draw.rect(s, (32, 40, 60), (0, 0, ww, self.TB),
                                 border_radius=8)
                pygame.draw.line(s, (70, 88, 128), (1, self.TB), (ww - 1, self.TB))
                if w.title_surf is None:
                    w.title_surf = self.app.font(13, bold=True).render(
                        w.title, True, (210, 220, 240))
                s.blit(w.title_surf, (8, 3))
                pygame.draw.circle(s, (90, 220, 130), (ww - 11, 11), 3)
                # содержимое
                s.set_clip((1, self.TB, ww - 2, hh - self.TB - 1))
                try:
                    w.draw(s, w.content(), self.app)
                finally:
                    s.set_clip(None)
                w._cache = s.copy()
                surface.blit(s, (x, y))
            else:
                surface.blit(cached, (x, y))
