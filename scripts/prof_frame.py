# -*- coding: utf-8 -*-
"""Профилировка кадра: что занимает время (update/render/подсистемы)."""
import os, sys, time
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["prof.py"]
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as m

app = m.App(type("A", (), {"selftest": True, "cam": 0, "width": None,
                           "height": None})())
app.keys = {"up": False, "down": False, "left": False, "right": False,
            "brake": False}

# прогрев
for _ in range(30):
    app.update(1 / 60.0)
    app.render()

pr = cProfile.Profile()
pr.enable()
for _ in range(180):
    app.update(1 / 60.0)
    app.render()
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(24)
print(s.getvalue()[:4200])
