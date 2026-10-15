"""P3 step 1: classify audio events -> dead-ball boundaries -> live segments.

Out: work/p3/segments.json  [{i, start, end, boundary: "start"|"whistle"|"buzzer"}]
"""
import json, os
import numpy as np

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfcfg import W, DUR
ev = json.load(open(f"{W}/audio/whistles.json"))   # from p3/whistle.py


def kind(e):
    if e["ton"] >= 55 and e["dur"] >= 0.8:
        return "buzzer"                      # scorer's table horn: timeout / period end / substitution
    if 3300 <= e["f"] <= 4500 and e["loud"] >= -37:
        return "whistle"                     # this court's referee (high-pitched whistle)
    if e["dur"] >= 0.3 or e["loud"] >= -38:
        return "weak"                        # second referee / adjacent court / squeak: low switch prior downstream
    return None


bounds = []
for e in ev:
    k = kind(e)
    if not k:
        continue
    if bounds and e["t"] - bounds[-1][0] < 3.0:     # one stoppage often has several blasts
        rank = {"weak": 0, "whistle": 1, "buzzer": 2}
        if rank[k] > rank[bounds[-1][1]]: bounds[-1] = (bounds[-1][0], k)
        continue
    bounds.append((e["t"], k))

cuts = [(0.0, "start")] + bounds
seg = [dict(i=i, start=round(t, 2), end=round(cuts[i + 1][0] if i + 1 < len(cuts) else DUR, 2), boundary=k)
       for i, (t, k) in enumerate(cuts)]
os.makedirs(f"{W}/p3", exist_ok=True)
json.dump(seg, open(f"{W}/p3/segments.json", "w"), indent=0)
L = np.array([s["end"] - s["start"] for s in seg])
print(f"events {len(ev)} -> whistles {sum(k == 'whistle' for _, k in bounds)}, weak {sum(k == 'weak' for _, k in bounds)}, buzzers {sum(k == 'buzzer' for _, k in bounds)} "
      f"-> {len(seg)} segments | length s: median {np.median(L):.0f}, max {L.max():.0f}")
for s in seg:
    print(f"  #{s['i']:2d} {int(s['start'])//60:02d}:{s['start']%60:04.1f}-{int(s['end'])//60:02d}:{s['end']%60:04.1f} ({s['end']-s['start']:5.1f}s) after {s['boundary']}")
