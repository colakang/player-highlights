"""Part-based colour descriptors: 5 horizontal body stripes x HSV hist (+ DINOv2 per stripe optional).
Out: work/p2/parts.npz  (same row order as emb.npz)"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
from pfcfg import W
import collections, csv, time
import cv2, numpy as np
Z = np.load(f"{W}/p2/emb.npz"); TR, FR = Z["track"], Z["frame"]
boxes = {}
for r in csv.DictReader(open(f"{W}/p1/tracks.csv")):
    boxes[(r["id"], int(r["frame"]))] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
STRIPES = [(0.0, 0.18), (0.18, 0.45), (0.45, 0.62), (0.62, 0.85), (0.85, 1.0)]   # head, torso, shorts, legs/knees, feet
def hist(c):
    h = cv2.calcHist([cv2.cvtColor(c, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return h / (h.sum() + 1e-6)
by_f = collections.defaultdict(list)
for i, (t, f) in enumerate(zip(TR, FR)): by_f[int(f)].append(i)
X = np.zeros((len(TR), len(STRIPES) * 192), np.float32); t0 = time.time()
for f, idx in by_f.items():
    im = cv2.imread(f"{W}/frames/f_{f:06d}.jpg")
    for i in idx:
        x1, y1, x2, y2 = boxes[(TR[i], f)]; h = y2 - y1; w = x2 - x1
        x1, x2 = int(max(0, x1 + 0.15 * w)), int(x2 - 0.15 * w)      # trim sides (background / neighbours)
        for k, (a, b) in enumerate(STRIPES):
            c = im[int(max(0, y1 + a * h)):int(y1 + b * h), x1:x2]
            if c.size: X[i, k * 192:(k + 1) * 192] = hist(c)
np.savez(f"{W}/p2/parts.npz", X=np.sqrt(X))   # Hellinger
print(f"parts {X.shape} in {time.time() - t0:.0f}s")
