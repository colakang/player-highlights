"""P4: #12 touch / action detection from her per-frame boxes (p4/identity.py -> her_boxes.json).

Cues per frame where her box is known:
  poss : a detector 'player-in-possession' / jump-shot / layup box overlaps her box (IoU >= --iou)
  ball : a ball detection inside her box (head..hips, widened by --near widths)
Events = runs of cue frames (gap <= --gap s), kept if >= --min_frames.
Out: {W}/p4/touches.json [{t, t_end, cues, frames}]

PF_WORK=... python p4/touch.py
"""
import argparse, collections, glob, json, os, pickle, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
from pfcfg import W, FPS

ap = argparse.ArgumentParser()
ap.add_argument("--iou", type=float, default=0.4); ap.add_argument("--near", type=float, default=0.6)
ap.add_argument("--gap", type=float, default=1.0); ap.add_argument("--min_frames", type=int, default=1)
ap.add_argument("--ball_conf", type=float, default=0.3); ap.add_argument("--poss_conf", type=float, default=0.3)
ap.add_argument("--out", default="touches.json"); ap.add_argument("--boxes", default="her_boxes.json"); ap.add_argument("--no_crop", action="store_true"); ap.add_argument("--owner", action="store_true", help="ball cue only when she is the closest player to it"); ap.add_argument("--veto", type=float, default=0, help="drop cues when the full-frame detector sees the ball (conf >= veto) more than 2 of her heights away"); ap.add_argument("--need_ball", action="store_true", help="events need >= 1 ball-cue frame (possession boxes alone are unreliable)"); ap.add_argument("--ytop", type=float, default=-0.1); ap.add_argument("--ybot", type=float, default=0.7)
a = ap.parse_args()

HB = {int(round(float(k) * FPS)): v for k, v in json.load(open(f"{W}/p4/{a.boxes}")).items()}
det = {}
for p in sorted(glob.glob(f"{W}/p1/det/chunk_*.pkl")): det.update(pickle.load(open(p, "rb")))
FULL = {f: [(b, c) for b, c, k in zip(d["xyxy"], d["conf"], d["cls"]) if k == "ball"] for f, d in det.items()}   # before crop merge
if os.path.exists(f"{W}/p4/ball_crop.pkl") and not a.no_crop:                     # zoomed re-detection around her (p4/ball_crop.py)
    for f, d in pickle.load(open(f"{W}/p4/ball_crop.pkl", "rb")).items():
        g = det.get(f) or dict(xyxy=[], conf=[], cls=[])
        det[f] = dict(xyxy=list(g["xyxy"]) + list(d["xyxy"]), conf=list(g["conf"]) + list(d["conf"]), cls=list(g["cls"]) + d["cls"])
POSS = {"player-in-possession", "player-jump-shot", "player-layup-dunk"}


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-6)


hits = []
PLAYER = {"player", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block"}


def hand_dist(cx, cy, b):
    """ball centre to a player's hands region (chest height, box-width units)"""
    hx, hy = (b[0] + b[2]) / 2, b[1] + 0.4 * (b[3] - b[1])
    return ((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5 / max(b[2] - b[0], 1)


for f, hb in sorted(HB.items()):
    d = det.get(f) or det.get(f + 1)
    if not d: continue
    x1, y1, x2, y2 = hb; w, h = x2 - x1, y2 - y1; cues = set()
    if a.veto and any(c >= a.veto and ((b[0] + b[2] - x1 - x2) ** 2 / 4 + (b[1] + b[3] - y1 - y2) ** 2 / 4) ** 0.5 > 2 * h
                      for g in (f - 1, f, f + 1) for b, c in FULL.get(g, [])): continue
    players = [b for b, c, k in zip(d["xyxy"], d["conf"], d["cls"]) if k in PLAYER and c >= 0.3]
    for b, c, k in zip(d["xyxy"], d["conf"], d["cls"]):
        if k in POSS and c >= a.poss_conf and iou(b, hb) >= a.iou: cues.add("poss")
        if k == "ball" and c >= a.ball_conf:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if x1 - a.near * w <= cx <= x2 + a.near * w and y1 + a.ytop * h <= cy <= y1 + a.ybot * h:
                # owner: the ball belongs to the closest player (hands height), not to her if a teammate / opponent is nearer
                if a.owner and any(iou(p, hb) < 0.5 and hand_dist(cx, cy, p) < hand_dist(cx, cy, hb) for p in players): continue
                cues.add("ball")
    if cues: hits.append((f, cues))
ev, cur = [], []
for f, c in hits:
    if cur and (f - cur[-1][0]) / FPS > a.gap: ev.append(cur); cur = []
    cur.append((f, c))
if cur: ev.append(cur)
out = [dict(t=e[0][0] / FPS, t_end=e[-1][0] / FPS, frames=len(e), cues=dict(collections.Counter(x for _, c in e for x in c)))
       for e in ev if len(e) >= a.min_frames and (not a.need_ball or any("ball" in c for _, c in e))]
os.makedirs(f"{W}/p4", exist_ok=True)
json.dump(out, open(f"{W}/p4/{a.out}", "w"), indent=0)
print(f"her frames {len(HB)} | cue frames {len(hits)} -> touch events {len(out)} "
      f"(poss {sum('poss' in e['cues'] for e in out)}, ball {sum('ball' in e['cues'] for e in out)})")
