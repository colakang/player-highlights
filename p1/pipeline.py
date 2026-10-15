"""P1 steps 2-5: track -> associate number boxes -> read -> vote -> segments + eval.

python p1/pipeline.py track|assoc|read|vote [--number 12 --color white]
All stages read/write under work/p1/.
"""
import argparse, collections, csv, glob, json, os, pickle, time
import cv2, numpy as np

import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
from pfcfg import W, FPS, DUR, GT
P = f"{W}/p1"
PLAYER_CLS = {"player", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block"}
NUM_CONF = 0.30
KNOWN = [] if W != "work" else [(20, 24), (60, 66), (486, 492), (563, 592), (598, 604), (786, 811), (821, 862)]  # user/SOP-confirmed #12


def load_dets():
    d = {}
    for f in sorted(glob.glob(f"{P}/det/chunk_*.pkl")):
        d.update(pickle.load(open(f, "rb")))
    return d


# ---------------------------------------------------------------- track
def stage_track(_):
    import supervision as sv
    from trackers import McByteTracker
    dets = load_dets()
    tracker = McByteTracker(frame_rate=FPS, lost_track_buffer=90, track_activation_threshold=0.4,
                            high_conf_det_threshold=0.4, enable_cmc=True, cmc_method="sparseOptFlow")
    rows, t = [], time.time()
    for fi in sorted(dets):
        d = dets[fi]
        keep = np.array([c in PLAYER_CLS for c in d["cls"]], bool)
        xyxy = d["xyxy"][keep] / 2  # tracker runs in 960x540
        sd = sv.Detections(xyxy=xyxy.astype(np.float32), confidence=d["conf"][keep], class_id=np.zeros(int(keep.sum()), int))
        frame = cv2.resize(cv2.imread(f"{W}/frames/f_{fi:06d}.jpg"), (960, 540))
        out = tracker.update(sd, frame)
        for b, tid in zip(out.xyxy * 2, out.tracker_id):
            if tid is not None and tid >= 0:
                rows.append([fi, int(tid), *[round(float(v), 1) for v in b]])
    with open(f"{P}/tracks.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "id", "x1", "y1", "x2", "y2"]); w.writerows(rows)
    print(f"tracks: {len({r[1] for r in rows})} ids, {len(rows)} boxes, {time.time() - t:.0f}s")


# ---------------------------------------------------------------- associate number boxes -> tracks
def ios(n, p):
    ix = max(0, min(n[2], p[2]) - max(n[0], p[0])); iy = max(0, min(n[3], p[3]) - max(n[1], p[1]))
    return ix * iy / max(1e-6, (n[2] - n[0]) * (n[3] - n[1]))


def bg_color(crop):
    """Median HSV of the jersey behind the digits (Otsu majority class)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg = m > 0 if (m > 0).mean() >= 0.5 else m == 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[bg]
    return [float(np.median(hsv[:, k])) for k in range(3)] if len(hsv) else [0, 0, 0]


def stage_assoc(_):
    dets = load_dets()
    by_frame = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{P}/tracks.csv")):
        by_frame[int(r["frame"])].append((r["id"], [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
    os.makedirs(f"{P}/numcrops", exist_ok=True)
    out, n_box, n_amb = [], 0, 0
    for fi in sorted(dets):
        d = dets[fi]
        nums = [(b, c) for b, c, k in zip(d["xyxy"], d["conf"], d["cls"]) if k == "number" and c >= NUM_CONF]
        if not nums:
            continue
        img = None
        for b, c in nums:
            n_box += 1
            ncx, ncy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            cands = []
            for tid, p in by_frame.get(fi, []):
                h = p[3] - p[1]
                if ios(b, p) >= 0.9 and p[1] <= ncy <= p[1] + 0.65 * h:
                    cands.append((abs(ncx - (p[0] + p[2]) / 2) / (p[2] - p[0]), tid))
            if not cands:
                continue
            cands.sort(); n_amb += len(cands) > 1
            if img is None:
                img = cv2.imread(f"{W}/frames/f_{fi:06d}.jpg")
            x1, y1, x2, y2 = map(int, b); px, py = int(0.15 * (x2 - x1)), int(0.15 * (y2 - y1))
            crop = img[max(0, y1 - py):y2 + py, max(0, x1 - px):x2 + px]
            path = f"{P}/numcrops/f{fi:06d}_{x1}_{y1}.png"
            cv2.imwrite(path, cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
            out.append(dict(frame=fi, track=cands[0][1], conf=float(c), box=[x1, y1, x2, y2],
                            ambiguous=len(cands) > 1, bg_hsv=bg_color(crop), crop=path))
    json.dump(out, open(f"{P}/assoc.json", "w"))
    print(f"number boxes>={NUM_CONF}: {n_box} | associated {len(out)} | ambiguous {n_amb} | tracks with >=1 box {len({o['track'] for o in out})}")


# ---------------------------------------------------------------- read
def stage_read(_):
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    MID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    A = json.load(open(f"{P}/assoc.json"))
    done = {}
    if os.path.exists(f"{P}/reads.jsonl"):
        for l in open(f"{P}/reads.jsonl"):
            r = json.loads(l); done[r["crop"]] = r["read"]
    model, proc = load(MID); cfg = load_config(MID)
    p = apply_chat_template(proc, cfg, "This is a crop of a jersey number. What number is it? Answer with digits only, or 'none'.", num_images=1)
    fh = open(f"{P}/reads.jsonl", "a"); t = time.time(); n = 0
    for a in A:
        if a["crop"] in done:
            continue
        o = generate(model, proc, p, [a["crop"]], max_tokens=6, verbose=False, temperature=0.0)
        fh.write(json.dumps(dict(crop=a["crop"], read=getattr(o, "text", str(o)).strip())) + "\n"); fh.flush()
        n += 1
        if n % 200 == 0:
            print(f"{n}/{len(A) - len(done)} {n / (time.time() - t):.1f}/s", flush=True)
    print("READ DONE", flush=True)


# ---------------------------------------------------------------- vote + segments + eval
COLOR_RULES = {  # jersey background colour in HSV (OpenCV ranges)
    "white": lambda h, s, v: s < 70 and v > 120,
    "teal": lambda h, s, v: 75 <= h <= 105 and s >= 70,
    "black": lambda h, s, v: v < 70,
}


def stage_vote(a):
    A = json.load(open(f"{P}/assoc.json"))
    R = {}
    for l in open(f"{P}/reads.jsonl"):
        r = json.loads(l); R[r["crop"]] = r["read"].lower().strip(" .")
    span = {}
    for r in csv.DictReader(open(f"{P}/tracks.csv")):
        f = int(r["frame"]); s = span.get(r["id"], (f, f)); span[r["id"]] = (min(s[0], f), max(s[1], f))
    per = collections.defaultdict(list)
    for x in A:
        rd = R.get(x["crop"])
        if rd is None:
            continue
        per[x["track"]].append((rd, COLOR_RULES[a.color](*x["bg_hsv"]), x["ambiguous"]))
    conf_t, weak_t = [], []
    for t, rs in per.items():
        valid = [(r, col) for r, col, amb in rs if r.isdigit() and not amb]
        hits = [col for r, col in valid if r == a.number]
        if not hits:
            continue
        color_ok = np.mean(hits) >= 0.5
        share = len(hits) / len(valid)
        if len(hits) >= 2 and share >= 0.5 and color_ok:
            conf_t.append(t)
        elif color_ok and share >= 0.5:
            weak_t.append(t)

    def segments(ts, gap=5, pad=2):
        iv = sorted((span[t][0] / FPS, span[t][1] / FPS) for t in ts); out = []
        for s, e in iv:
            if out and s - out[-1][1] < gap: out[-1][1] = max(out[-1][1], e)
            else: out.append([s, e])
        return [[max(0, s - pad), min(DUR, e + pad)] for s, e in out]

    def cover(segs):
        return sum(sum(max(0, min(e, b) - max(s, a_)) for a_, b in segs) for s, e in KNOWN)

    ts = lambda x: f"{int(x) // 60:02d}:{int(x % 60):02d}"
    tracks_w_reads = len(per)
    for name, tl in (("confirmed", conf_t), ("confirmed+weak", conf_t + weak_t)):
        sg = segments(tl)
        print(f"[{name}] tracks {len(tl)} | segments {len(sg)} | total {sum(e - s for s, e in sg):.0f}s | known coverage {cover(sg):.0f}/117s")
    sg = segments(conf_t + weak_t)
    json.dump(dict(confirmed=conf_t, weak=weak_t, segments=sg), open(f"{P}/timeline.json", "w"), indent=1)
    print(f"tracks with any read: {tracks_w_reads}")
    for s, e in sg:
        print(f"  {ts(s)} – {ts(e)}  ({e - s:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["track", "assoc", "read", "vote"])
    ap.add_argument("--number", default="12"); ap.add_argument("--color", default="white")
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)


# ---------------------------------------------------------------- adaptive team colour (per video)
COLOR_PROTO = {  # rough HSV prototypes (OpenCV ranges) for the colour name the user gives
    "white": (0, 20, 220), "black": (0, 30, 40), "grey": (0, 20, 130), "teal": (90, 150, 120), "red": (0, 200, 170),
    "blue": (110, 180, 150), "navy": (115, 160, 70), "yellow": (25, 180, 200), "green": (60, 150, 120),
    "orange": (12, 200, 220), "purple": (140, 140, 120), "pink": (165, 120, 220),
}


def _hsv_feat(h):
    h = np.asarray(h, float); ang = h[..., 0] / 180 * 2 * np.pi
    return np.stack([np.cos(ang) * h[..., 1] / 255, np.sin(ang) * h[..., 1] / 255, h[..., 1] / 255, h[..., 2] / 255], -1)


def team_color_fn(color, k=3):
    """Cluster all number-box backgrounds of this video, keep the cluster nearest the named colour.
    Robust to gym lighting (e.g. white reads as S~90 under warm light). Cached in {P}/teamcolor_<color>.json."""
    cache = f"{P}/teamcolor_{color}.json"
    if os.path.exists(cache):
        c = json.load(open(cache)); cen = np.array(c["centers"]); keep = c["keep"]
    else:
        from sklearn.cluster import KMeans
        H = np.array([x["bg_hsv"] for x in json.load(open(f"{P}/assoc.json"))], float)
        km = KMeans(k, n_init=10, random_state=0).fit(_hsv_feat(H)); cen = km.cluster_centers_
        keep = int(np.argmin(np.linalg.norm(cen - _hsv_feat(COLOR_PROTO[color]), axis=1)))
        json.dump(dict(centers=cen.tolist(), keep=keep), open(cache, "w"))
    return lambda h, s, v: int(np.argmin(np.linalg.norm(cen - _hsv_feat((h, s, v)), axis=1))) == keep
