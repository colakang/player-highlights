"""Segment check of the target's boxes with the game ReID scores (GTA-Link idea: verify whole tracklets, not frames).

Her boxes are cut into continuous segments (frame gap <= 2 and boxes overlapping); a segment whose median ReID score
(p4/reid_score.py) is below --tau is dropped - typically the tracking carried on onto a teammate, an opponent or empty
floor after she left the picture. On three hand-checked games (leave-one-game-out): precision 77 -> 86 %, accuracy 73 -> 71 %.

PF_WORK=<period work dir> python p4/reid_segments.py [--tau 0.5]
"""
import argparse, json, os, shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tau", type=float, default=0.5)
a = ap.parse_args()
W = os.environ["PF_WORK"].rstrip("/"); FPS = 6


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


base = f"{W}/p4/her_boxes_presegment.json"
if not os.path.exists(base): shutil.copy(f"{W}/p4/her_boxes.json", base)
HB = {int(round(float(t) * FPS)): b for t, b in json.load(open(base)).items()}
S = {int(k): v for k, v in json.load(open(f"{W}/p4/reid_scores.json")).items()}
sc = {}
for f, b in HB.items():
    s = [x[4] for x in S.get(f, []) if iou(x[:4], b) >= 0.5]; sc[f] = max(s) if s else None
segs, cur = [], []
for f in sorted(HB):
    if cur and (f - cur[-1] > 2 or iou(HB[f], HB[cur[-1]]) < 0.1): segs.append(cur); cur = []
    cur.append(f)
if cur: segs.append(cur)
out, dropped = {}, 0
for s in segs:
    v = [sc[f] for f in s if sc[f] is not None]
    if v and np.median(v) < a.tau: dropped += len(s); continue
    for f in s: out[f] = HB[f]
json.dump({f"{f / FPS:.3f}": b for f, b in sorted(out.items())}, open(f"{W}/p4/her_boxes.json", "w"))
print(f"[reid_segments] {len(segs)} segments, dropped {dropped} of {len(HB)} frames")
