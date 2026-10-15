"""Vision-only dead-ball boundaries (venues where the referee whistle is not usable in the audio).

A boundary = start of a sustained slowdown: >= --play s of play (mean moving players > --hi) followed by
>= --quiet s with smoothed moving players < --lo. Written in whistles.json format so p3/segment.py / oncourt.py work unchanged.

PF_WORK=... python p3/boundaries_vision.py [--hi 1.5 --lo 0.6 --quiet 3 --play 5]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfcfg import W

ap = argparse.ArgumentParser()
ap.add_argument("--hi", type=float, default=1.5); ap.add_argument("--lo", type=float, default=0.6)
ap.add_argument("--quiet", type=int, default=3); ap.add_argument("--play", type=int, default=5)
ap.add_argument("--gap", type=float, default=8.0); ap.add_argument("--out", default="whistles.json")
ap.add_argument("--grid", type=float, default=10.0, help="also add a weak boundary every N s (0 = off); validated: as good as whistles")
a = ap.parse_args()
act = np.array(json.load(open(f"{W}/p3/activity.json"))["moving"])
sm = np.convolve(act, np.ones(3) / 3, "same"); ev = []
for s in range(a.play, len(sm) - a.quiet):
    if sm[s - a.play:s].mean() > a.hi and (sm[s:s + a.quiet] < a.lo).all() and (not ev or s - ev[-1]["t"] > a.gap):
        ev.append(dict(t=float(s), dur=0.5, f=3800.0, ton=40.0, loud=-30.0, src="vision"))
nv = len(ev)
if a.grid > 0:
    ev += [dict(t=float(g), dur=0.3, f=2600.0, ton=30.0, loud=-38.0, src="grid") for g in np.arange(a.grid, len(sm), a.grid)
           if all(abs(g - e["t"]) > 4 for e in ev[:nv])]
ev.sort(key=lambda e: e["t"])
json.dump(ev, open(f"{W}/audio/{a.out}", "w"), indent=0)
print(f"vision boundaries: {nv} + grid {len(ev) - nv} over {len(sm)} s")
