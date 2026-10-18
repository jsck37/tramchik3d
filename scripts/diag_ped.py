# -*- coding: utf-8 -*-
"""Трассировка пешехода 0 и состава в момент зависания."""
import math
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as m  # noqa: E402


class A:
    selftest = True
    cam = None
    width = None
    height = None


app = m.App(A())
app.keys = {"up": False, "down": False, "left": False, "right": False,
            "brake": False}
app.tram.reset()
app.tram.auto = True

# едем до зависания: крутим до 60 с, потом трассируем
for _ in range(3600):
    app.update(1 / 60)

ped = app.world.pedestrians[0]
print("трам: pos", app.tram.pos.round(2), "yaw", round(app.tram.yaw, 2),
      "speed", round(app.tram.speed, 2))
print("сегменты:", [(round(x0, 2), round(z0, 2), round(x1, 2), round(z1, 2), hw)
                    for (x0, z0, x1, z1, hw) in app.tram.segments()])
print("ped0: pos", ped.pos.round(3), "target", ped.target.round(2),
      "knocked", round(ped.knocked, 2), "yaw", round(ped.yaw, 2))
for i in range(12):
    app.update(1 / 60)
    print(f"f{i:02d} ped0 {ped.pos.round(3)} speed {ped.speed:.2f} "
          f"tram {app.tram.pos.round(2)} v{app.tram.speed:.2f}")
