"""Apply game-specific ReID scores (p4/reid_score.py) to the target's boxes with fixed decision rules.

Per frame, using every tracked non-opponent box's cosine score to the #target prototype:
  keep the current her-box if its score >= --keep;
  else replace it by the best box if best >= --add and best - second >= --margin;
  else drop it if its score <= --drop;
  with no her-box, add the best box under the same --add / --margin test.
Defaults were chosen on three hand-checked games (accuracy 65 -> 74 %, precision 75 -> 81 % on "which box is the
player"; leave-one-game-out 65 -> 72 % / 75 -> 76 %).

PF_WORK=<period work dir> python p4/reid_apply.py
Out: {W}/p4/her_boxes.json (previous kept as her_boxes_prereid.json)
"""
import argparse, json, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--keep", type=float, default=0.6); ap.add_argument("--add", type=float, default=0.8)
ap.add_argument("--margin", type=float, default=0.1); ap.add_argument("--drop", type=float, default=0.3)
a = ap.parse_args()
W = os.environ["PF_WORK"].rstrip("/"); FPS = 6


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


base = f"{W}/p4/her_boxes_prereid.json"
if not os.path.exists(base): shutil.copy(f"{W}/p4/her_boxes.json", base)
HB = {int(round(float(t) * FPS)): b for t, b in json.load(open(base)).items()}
S = {int(f): v for f, v in json.load(open(f"{W}/p4/reid_scores.json")).items()}
out, n = {}, dict(kept=0, replaced=0, dropped=0, added=0)
for f in sorted(set(S) | set(HB)):
    L = sorted(S.get(f, []), key=lambda x: -x[4]); cur = HB.get(f)
    s1 = L[0][4] if L else -1; s2 = L[1][4] if len(L) > 1 else -1; b1 = L[0][:4] if L else None
    strong = bool(L) and s1 >= a.add and s1 - s2 >= a.margin
    if cur is not None:
        cs = [x[4] for x in L if iou(x[:4], cur) >= 0.5]; c = max(cs) if cs else None
        if c is not None and c >= a.keep: out[f] = cur; n["kept"] += 1
        elif strong and iou(b1, cur) < 0.3: out[f] = b1; n["replaced"] += 1
        elif c is not None and c <= a.drop: n["dropped"] += 1
        else: out[f] = cur; n["kept"] += 1
    elif strong: out[f] = b1; n["added"] += 1
json.dump({f"{f / FPS:.3f}": [float(x) for x in b] for f, b in sorted(out.items())}, open(f"{W}/p4/her_boxes.json", "w"))
print(f"[reid_apply] {n} | her frames {len(HB)} -> {len(out)}")
