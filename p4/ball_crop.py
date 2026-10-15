"""P4: re-detect the ball (and possession poses) on a zoomed crop around the target's box.

On the full 1080p frame squeezed into the 1280 detector input the ball is ~25 px and mostly missed; a square crop of
--k x her height (clamped to [--smin, 1080]) upscaled to 1280 makes it 3-4x larger. Only frames with her box.
Out: {W}/p4/ball_crop.pkl {frame: {"xyxy", "conf", "cls"}} in full-frame coordinates (touch.py merges it).

PF_WORK=... python p4/ball_crop.py
"""
import argparse, json, os, pickle, sys, time
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "p1")]
from pfcfg import W, FPS
from rfdet import YoloOnnx

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=float, default=3.0); ap.add_argument("--smin", type=int, default=480)
ap.add_argument("--conf", type=float, default=0.15)
a = ap.parse_args()
KEEP = {"ball", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block"}
HB = {int(round(float(t) * FPS)): b for t, b in json.load(open(f"{W}/p4/her_boxes.json")).items()}
model = YoloOnnx(); out = {}; t0 = time.time()
for f, (x1, y1, x2, y2) in sorted(HB.items()):
    im = cv2.imread(f"{W}/frames/f_{f:06d}.jpg")
    if im is None: continue
    H, Wd = im.shape[:2]
    s = int(min(H, max(a.smin, a.k * (y2 - y1))))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ox = int(min(max(0, cx - s / 2), Wd - s)); oy = int(min(max(0, cy - s / 2), H - s))
    xyxy, conf, cls = model(im[oy:oy + s, ox:ox + s], conf=a.conf, iou=0.9)
    k = [i for i, c in enumerate(cls) if c in KEEP]
    out[f] = dict(xyxy=xyxy[k] + np.array([ox, oy, ox, oy], np.float32), conf=conf[k], cls=[cls[i] for i in k])
pickle.dump(out, open(f"{W}/p4/ball_crop.pkl", "wb"))
nb = sum("ball" in d["cls"] for d in out.values())
print(f"[ball_crop] {len(out)} frames | ball found in {nb} ({nb / max(1, len(out)):.0%}) | {time.time() - t0:.0f} s")
