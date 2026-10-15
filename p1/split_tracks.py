"""Split McByte tracks at identity changes (ID switches) into cleaner tracklets.

Per box: jersey team colour of the torso (teal / white / other) and a DINOv2 embedding (sampled).
A track is cut where (a) the smoothed team colour flips between teams for >= --run frames, or
(b) the appearance embedding jumps (cosine distance between the means of the --win frames before and after > --jump).
Rewrites {W}/p1/tracks.csv (original kept as tracks_raw.csv) with ids "<track>_<k>", and remaps assoc.json.
Downstream (recheck / oncourt / refine / her_tracks ...) must be re-run afterwards.

PF_WORK=... python p1/split_tracks.py [--no-embed]
"""
import argparse, collections, csv, json, os, shutil, sys, time
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfcfg import W

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=int, default=3, help="frames a colour change must persist")
ap.add_argument("--win", type=int, default=4); ap.add_argument("--jump", type=float, default=0.35)
ap.add_argument("--no-embed", action="store_true"); ap.add_argument("--min_len", type=int, default=2)
ap.add_argument("--color", default="white")
a = ap.parse_args()

src = f"{W}/p1/tracks_raw.csv" if os.path.exists(f"{W}/p1/tracks_raw.csv") else f"{W}/p1/tracks.csv"
if not os.path.exists(f"{W}/p1/tracks_raw.csv"): shutil.copy(f"{W}/p1/tracks.csv", f"{W}/p1/tracks_raw.csv")
rows = list(csv.DictReader(open(src)))
by_f = collections.defaultdict(list)
for i, r in enumerate(rows): by_f[int(r["frame"])].append(i)

# jersey colours anchored on this video's own number-box backgrounds (p1/pipeline.py team_color_fn clusters):
# target team = cluster nearest the named colour, opponent = the most frequent other cluster. Works for any opponent colour.
def _feat(h):
    h = np.asarray(h, float); ang = h[..., 0] / 180 * 2 * np.pi
    return np.stack([np.cos(ang) * h[..., 1] / 255, np.sin(ang) * h[..., 1] / 255, h[..., 1] / 255, h[..., 2] / 255], -1)
_tc = json.load(open(f"{W}/p1/teamcolor_{a.color}.json")); CEN = np.array(_tc["centers"]); KEEP = _tc["keep"]
_bg = np.array([x["bg_hsv"] for x in json.load(open(f"{W}/p1/assoc_raw.json" if os.path.exists(f"{W}/p1/assoc_raw.json") else f"{W}/p1/assoc.json"))], float)
_cnt = np.bincount(np.argmin(((_feat(_bg)[:, None] - CEN[None]) ** 2).sum(-1), 1), minlength=len(CEN))
# opponent = every other coloured cluster (the opponent colour often splits into two shades, e.g. teal 111/161 sat)
OPP = [k for k in range(len(CEN)) if k != KEEP and CEN[k][2] > 0.3] or [max((k for k in range(len(CEN)) if k != KEEP), key=lambda k: _cnt[k])]


def jersey(torso_hsv):
    near = np.argmin(((_feat(torso_hsv.reshape(-1, 3))[:, None] - CEN[None]) ** 2).sum(-1), 1)
    ft, fo = (near == KEEP).mean(), np.isin(near, OPP).mean()
    return 2 if ft > 0.45 and ft > 2 * fo else 1 if fo > 0.45 and fo > 2 * ft else 0


col = np.zeros(len(rows), int)          # 1 = teal, 2 = white, 0 = unknown
crops = {}
t0 = time.time()
for f, idx in by_f.items():
    im = cv2.imread(f"{W}/frames/f_{f:06d}.jpg")
    if im is None: continue
    for i in idx:
        x1, y1, x2, y2 = [float(rows[i][k]) for k in ("x1", "y1", "x2", "y2")]
        c = im[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        if c.size == 0 or c.shape[0] < 24: continue
        col[i] = jersey(cv2.cvtColor(c[int(.2 * len(c)):int(.55 * len(c)), int(.25 * c.shape[1]):int(.75 * c.shape[1])], cv2.COLOR_BGR2HSV))
        if not a.no_embed: crops[i] = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (112, 224))
print(f"colour per box: {len(rows)} boxes in {time.time() - t0:.0f}s | opponent {(col == 1).sum()} target-team {(col == 2).sum()} unknown {(col == 0).sum()}", flush=True)

emb = {}
if not a.no_embed:
    import torch
    DEV = "mps"; m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(DEV)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    keys = list(crops); t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(keys), 256):
            b = np.stack([crops[k] for k in keys[i:i + 256]]).astype(np.float32) / 255
            x = torch.from_numpy((b - mean) / std).permute(0, 3, 1, 2).float().to(DEV)
            e = torch.nn.functional.normalize(m(x), dim=1).cpu().numpy()
            for k, v in zip(keys[i:i + 256], e): emb[k] = v
    print(f"embeddings {len(emb)} in {time.time() - t0:.0f}s", flush=True)

tracks = collections.defaultdict(list)
for i, r in enumerate(rows): tracks[r["id"]].append(i)
new_id = {}; n_cut = collections.Counter()
for t, idx in tracks.items():
    idx.sort(key=lambda i: int(rows[i]["frame"]))
    cuts = set()
    # (a) team colour flips (persisting >= run frames)
    known = [(k, col[i]) for k, i in enumerate(idx) if col[i]]
    cur, run_start, run_len = None, None, 0
    for k, c in known:
        if cur is None: cur = c; continue
        if c != cur:
            if run_len == 0: run_start = k
            run_len += 1
            if run_len >= a.run: cuts.add(run_start); n_cut["colour"] += 1; cur = c; run_len = 0
        else: run_len = 0
    # (b) appearance jumps
    if emb:
        E = [(k, emb[i]) for k, i in enumerate(idx) if i in emb]
        for j in range(a.win, len(E) - a.win):
            if any(abs(E[j][0] - c) <= a.win for c in cuts): continue
            before = np.mean([e for _, e in E[j - a.win:j]], 0); after = np.mean([e for _, e in E[j:j + a.win]], 0)
            d = 1 - before @ after / (np.linalg.norm(before) * np.linalg.norm(after) + 1e-9)
            if d > a.jump:
                local = [1 - np.mean([e for _, e in E[q - a.win:q]], 0) @ np.mean([e for _, e in E[q:q + a.win]], 0) for q in (j - 1, j + 1) if a.win <= q < len(E) - a.win]
                if all(d >= x - 1e-9 for x in local): cuts.add(E[j][0]); n_cut["appearance"] += 1
    seg = 0
    for k, i in enumerate(idx):
        if k in cuts: seg += 1
        new_id[i] = f"{t}_{seg}"
with open(f"{W}/p1/tracks.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "id", "x1", "y1", "x2", "y2"])
    for i, r in enumerate(rows): w.writerow([r["frame"], new_id[i], r["x1"], r["y1"], r["x2"], r["y2"]])
# remap assoc (number box -> track) by (track, frame)
key = {(r["id"], int(r["frame"])): new_id[i] for i, r in enumerate(rows)}
if not os.path.exists(f"{W}/p1/assoc_raw.json"): shutil.copy(f"{W}/p1/assoc.json", f"{W}/p1/assoc_raw.json")
A = json.load(open(f"{W}/p1/assoc_raw.json"))
ri = {(r["id"], int(r["frame"])): i for i, r in enumerate(rows)}
n_drop = 0
for x in A:
    i = ri.get((x["track"], x["frame"]))
    x["track"] = key.get((x["track"], x["frame"]), x["track"])
    # a number on a white background cannot belong to a box whose jersey is teal (the number box fell into the
    # defender's box while the real owner was not detected separately)
    x["box_colour"] = int(col[i]) if i is not None else 0
    bg_team = int(np.argmin(((_feat(np.array(x["bg_hsv"], float))[None] - CEN) ** 2).sum(-1))) == KEEP
    if x["box_colour"] == 1 and bg_team: x["colour_conflict"] = True; n_drop += 1
A = [x for x in A if not x.get("colour_conflict")]
print(f"number boxes dropped for jersey-colour conflict: {n_drop}")
json.dump(A, open(f"{W}/p1/assoc.json", "w"))
json.dump(dict(colour={rows[i]["id"] + "@" + rows[i]["frame"]: int(col[i]) for i in range(len(rows)) if col[i]}),
          open(f"{W}/p1/box_colour.json", "w"))
L = collections.Counter(new_id.values())
print(f"tracks {len(tracks)} -> tracklets {len(L)} | cuts {dict(n_cut)} | median len {np.median(list(L.values())) / 6:.1f}s")
