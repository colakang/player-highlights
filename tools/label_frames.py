"""Fine-tuning step 1: pick frames from your analysed games for labelling (images stay on this Mac).

python tools/label_frames.py --games work/<game1> work/<game2> ... [--test work/<game3>] [--n 40] [--out work/labels]

Per game --n frames from on-court time, spread out in time (>= --gap s apart): ~3/4 where the current detector is
unsure (weak ball, crowded players, far court, unsure numbers), ~1/4 random. Frames come straight from the source
video at full resolution. Games given with --test are kept apart for measuring (never trained on).
Out: <out>/images/*.jpg, <out>/frames.json
"""
import argparse, glob, json, os, pickle, random, re, subprocess
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--games", nargs="+", required=True, help="work dirs made by pf.py (work/<game>)")
ap.add_argument("--test", nargs="*", default=[], help="work dirs used only for testing")
ap.add_argument("--out", default="work/labels"); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--gap", type=float, default=5.0); ap.add_argument("--hard", type=float, default=0.75); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed); FPS = 6.0
os.makedirs(f"{a.out}/images", exist_ok=True)
PLAYER = ("player", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block")


def hardness(d):
    k = list(d["cls"]); c = np.asarray(d["conf"]); b = np.asarray(d["xyxy"]).reshape(-1, 4); s = 0.0
    ball = [ci for ci, ki in zip(c, k) if ki == "ball"]
    if ball and max(ball) < 0.45: s += 2                                  # unsure ball only
    if not ball: s += 0.5
    pl = b[[ki in PLAYER for ki in k]]
    if len(pl):
        if np.median(pl[:, 3] - pl[:, 1]) < 150: s += 1                    # far court
        cx = (pl[:, 0] + pl[:, 2]) / 2
        s += min(2, sum(1 for i in range(len(cx)) for j in range(i + 1, len(cx)) if abs(cx[i] - cx[j]) < 40) / 2)   # crowding
    nums = [ci for ci, ki in zip(c, k) if ki == "number"]
    if nums and min(nums) < 0.5: s += 1
    return s


picked = []
for G in a.games + a.test:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.basename(os.path.normpath(G))); cands = []
    for W in sorted(glob.glob(f"{G}/*/p3/timeline.json")):
        W = os.path.dirname(os.path.dirname(W)); P = os.path.basename(W).upper()
        if P.endswith("_ID"): continue                                 # identity sub-dir, same frames
        det = {}
        for p in sorted(glob.glob(f"{W}/p1/det/chunk_*.pkl")): det.update(pickle.load(open(p, "rb")))
        on = json.load(open(f"{W}/p3/timeline.json"))["intervals"]; video = os.path.realpath(f"{W}/game.mp4")
        cands += [(hardness(d), P, f, video) for f, d in det.items() if any(s * FPS <= f <= e * FPS for s, e in on)]
    if not cands: print(f"{G}: no analysed periods found (run pf.py first)"); continue
    random.shuffle(cands); chosen = []
    ok = lambda P, f: all(not (q == P and abs(f - g) < a.gap * FPS) for _, q, g, _ in chosen)
    for c in sorted(cands, key=lambda x: -x[0]):
        if len(chosen) >= int(round(a.n * a.hard)): break
        if ok(c[1], c[2]): chosen.append(c)
    for c in cands:
        if len(chosen) >= a.n: break
        if ok(c[1], c[2]): chosen.append(c)
    split = "test" if G in a.test else "train"
    for h, P, f, video in chosen:
        img = f"{a.out}/images/{name}_{P}_{f:06d}.jpg"
        if not os.path.exists(img):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{(f - 1) / FPS:.3f}", "-i", video, "-frames:v", "1", "-q:v", "2", img], check=True)
        picked.append(dict(image=os.path.basename(img), game=name, period=P, frame=f, split=split))
    print(f"{name}: {len(chosen)} frames ({split})", flush=True)
json.dump(picked, open(f"{a.out}/frames.json", "w"), indent=0)
print(f"{len(picked)} frames -> {a.out}/images")
