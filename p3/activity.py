"""Per-second play activity from P1 tracks: how many players are moving, normalised by their box height."""
import collections, csv, json
import numpy as np
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfcfg import W, FPS, DUR, GT
tr = collections.defaultdict(dict)
for r in csv.DictReader(open(f"{W}/p1/tracks.csv")):
    tr[r["id"]][int(r["frame"])] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
nsec = int(max(np.ceil(DUR), max(max(fb) for fb in tr.values()) / FPS + 1)); moving = np.zeros(nsec); present = np.zeros(nsec)
for t, fb in tr.items():
    fs = sorted(fb)
    for a, b in zip(fs, fs[1:]):
        if b - a != 1: continue
        p, q = np.array(fb[a]), np.array(fb[b]); h = p[3] - p[1]
        if h < 60: continue                                   # tiny far-away boxes are unreliable
        v = np.hypot(*(((q[:2] + q[2:]) - (p[:2] + p[2:])) / 2)) / h * FPS   # body-heights per second
        s = int(a / FPS); present[s] += 1 / FPS
        if v > 0.5: moving[s] += 1 / FPS
act = moving                                                  # ~ number of players running each second
os.makedirs(f"{W}/p3", exist_ok=True)
json.dump(dict(moving=moving.round(2).tolist(), present=present.round(2).tolist()), open(f"{W}/p3/activity.json", "w"))
if not GT: raise SystemExit
g = json.load(open(GT))["intervals"]
st = np.array([""] * nsec, object)
for x in g: st[int(x["start"]):int(np.ceil(x["end"]))] = x["state"]
for k in ("on", "off", "dead"):
    v = act[st == k]; print(f"{k:4s}: moving players/s  p10 {np.percentile(v,10):.1f}  p25 {np.percentile(v,25):.1f}  median {np.median(v):.1f}  p75 {np.percentile(v,75):.1f}")
sm = np.convolve(act, np.ones(9) / 9, "same")
print("9s-smoothed:", {k: f"median {np.median(sm[st == k]):.2f} p90 {np.percentile(sm[st == k], 90):.2f}" for k in ("on", "off", "dead")})
