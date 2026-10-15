"""P4-A: shot-attempt candidates inside the player's on-court time (recall-first; a human confirms).

Cues (any of them, merged within --merge seconds):
  rise : the ball moves up fast and ends above the heads of the players around it
  pose : detector 'player-jump-shot' / 'player-layup-dunk' (noisy on this footage, kept for recall)
  rim  : ball inside the area around a detected rim
Each candidate also lists the white tracks nearest the ball before the cue, flagging confirmed #target tracks.

Out: {W}/p4/shot_cands.json
"""
import argparse, collections, csv, glob, json, os, pickle, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
from pfcfg import W, FPS

ap = argparse.ArgumentParser()
ap.add_argument("--merge", type=float, default=3.0); ap.add_argument("--ball_conf", type=float, default=0.3)
a = ap.parse_args()

det = {}
for f in sorted(glob.glob(f"{W}/p1/det/chunk_*.pkl")): det.update(pickle.load(open(f, "rb")))
on = json.load(open(f"{W}/p3/timeline.json"))["intervals"]
in_on = lambda t: any(s - 1 <= t <= e + 1 for s, e in on)
her = set(json.load(open(f"{W}/p4/her_tracks.json"))["confirmed"]) if os.path.exists(f"{W}/p4/her_tracks.json") else set()
tracks = collections.defaultdict(list)
for r in csv.DictReader(open(f"{W}/p1/tracks.csv")):
    tracks[int(r["frame"])].append((r["id"], [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))

PLAYERS = {"player", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block"}
ball, cues = {}, []
for f in sorted(det):
    d = det[f]; bs = [(b, c) for b, c, k in zip(d["xyxy"], d["conf"], d["cls"]) if k == "ball" and c >= a.ball_conf]
    if bs: ball[f] = max(bs, key=lambda x: x[1])[0]
    for b, c, k in zip(d["xyxy"], d["conf"], d["cls"]):
        if k in ("player-jump-shot", "player-layup-dunk") and c >= 0.3: cues.append((f, "pose"))
        if k == "rim" and f in ball:
            bx = ball[f]; cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2; w, h = b[2] - b[0], b[3] - b[1]
            if b[0] - w <= cx <= b[2] + w and b[1] - 2 * h <= cy <= b[3] + h: cues.append((f, "rim"))

fs = sorted(ball)
for i, f in enumerate(fs):                       # rise: within 1 s the ball climbs > 0.6 player-heights and ends above heads
    for g in fs[i + 1:i + 8]:
        if g - f > FPS: break
        b0, b1 = ball[f], ball[g]; y0, y1 = (b0[1] + b0[3]) / 2, (b1[1] + b1[3]) / 2
        ps = [p for k, p, c in zip(det[g]["cls"], det[g]["xyxy"], det[g]["conf"]) if k in PLAYERS and c >= 0.3]
        if not ps: continue
        near = sorted(ps, key=lambda p: abs((p[0] + p[2]) / 2 - (b1[0] + b1[2]) / 2))[:3]
        ph = np.median([p[3] - p[1] for p in near]); top = min(p[1] for p in near)
        if y0 - y1 > 0.6 * ph and y1 < top:
            cues.append((g, "rise")); break

cues = sorted((f, k) for f, k in cues if in_on(f / FPS))
cands, cur = [], []
for f, k in cues:
    if cur and (f - cur[-1][0]) / FPS > a.merge: cands.append(cur); cur = []
    cur.append((f, k))
if cur: cands.append(cur)


def shooter_tracks(f0):
    """white-ish tracks closest to the ball in the 1.5 s before the cue."""
    near = collections.Counter()
    for g in range(max(1, f0 - int(1.5 * FPS)), f0 + 1):
        if g not in ball: continue
        b = ball[g]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        best = min(tracks.get(g, []), key=lambda t: np.hypot((t[1][0] + t[1][2]) / 2 - cx, t[1][1] - cy), default=None)
        if best: near[best[0]] += 1
    return [t for t, _ in near.most_common(3)]


out = []
for c in cands:
    f0 = c[0][0]; sh = shooter_tracks(f0)
    out.append(dict(t=f0 / FPS, t_end=c[-1][0] / FPS, cues=dict(collections.Counter(k for _, k in c)), near_tracks=sh,
                    her_near=bool(set(sh) & her)))
os.makedirs(f"{W}/p4", exist_ok=True)
json.dump(out, open(f"{W}/p4/shot_cands.json", "w"), indent=0)
print(f"cues {len(cues)} -> candidates {len(out)} (her confirmed track near ball: {sum(o['her_near'] for o in out)}) | by cue:",
      dict(collections.Counter(k for o in out for k in o["cues"])))
