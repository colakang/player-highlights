"""Tracking with BoxMOT (appearance-aware association, OSNet ReID on MPS) on the existing detections.

PF_WORK=... python p1/track_boxmot.py [--tracker botsort|deepocsort|strongsort] [--reid osnet_x0_25_msmt17.pt]
Writes {W}/p1/tracks.csv in the same format as p1/pipeline.py track (frame, id, x1, y1, x2, y2).
"""
import argparse, csv, glob, os, pickle, sys, time
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfcfg import W, FPS

ap = argparse.ArgumentParser()
ap.add_argument("--tracker", default="botsort", choices=["botsort", "deepocsort", "strongsort"])
ap.add_argument("--reid", default="osnet_x0_25_msmt17.pt"); ap.add_argument("--device", default="mps")
ap.add_argument("--out", default="tracks.csv")
a = ap.parse_args()
PLAYER = {"player", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block"}
import boxmot
reid = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", a.reid))
reid.parent.mkdir(exist_ok=True)
common = dict(reid_weights=reid, device=a.device, half=False)
if a.tracker == "botsort":
    T = boxmot.BotSort(track_high_thresh=0.4, track_low_thresh=0.1, new_track_thresh=0.5, track_buffer=90, frame_rate=int(FPS),
                       appearance_thresh=0.25, proximity_thresh=0.5, cmc_method="ecc", **common)
elif a.tracker == "deepocsort":
    T = boxmot.DeepOcSort(**common)
else:
    T = boxmot.StrongSort(**common)
det = {}
for f in sorted(glob.glob(f"{W}/p1/det/chunk_*.pkl")): det.update(pickle.load(open(f, "rb")))
rows, t0 = [], time.time()
for k, fi in enumerate(sorted(det)):
    d = det[fi]; m = np.array([c in PLAYER for c in d["cls"]], bool)
    D = np.c_[d["xyxy"][m], d["conf"][m], np.zeros(int(m.sum()))].astype(np.float32) if m.any() else np.zeros((0, 6), np.float32)
    img = cv2.imread(f"{W}/frames/f_{fi:06d}.jpg")
    out = T.update(D, img)
    if hasattr(out, "xyxy"): out = np.c_[out.xyxy, out.tracker_id]
    for o in np.asarray(out):
        rows.append([fi, int(o[4]), *[round(float(v), 1) for v in o[:4]]])
    if k % 1000 == 0: print(f"{k} frames {time.time() - t0:.0f}s", flush=True)
with open(f"{W}/p1/{a.out}", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "id", "x1", "y1", "x2", "y2"]); w.writerows(rows)
print(f"{a.tracker}: {len({r[1] for r in rows})} tracks, {len(rows)} boxes, {time.time() - t0:.0f}s ({len(det) / (time.time() - t0):.1f} fps)")
