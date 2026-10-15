"""P2 step 1: per-box appearance embeddings (DINOv2-S) + per-track team probability (SigLIP zero-shot from colour text).

Out: work/p2/emb.npz  track[], frame[], emb[N,384]
     work/p2/team.json {track: {"p_target_team": float, "probs": {...}}}
"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
from pfcfg import W
import argparse, collections, csv, json, os, time
import cv2, numpy as np, torch

DEV = "mps"
PER_TRACK = 24          # boxes sampled per track for embeddings
TEAM_CROPS = 4          # boxes per track for team vote
IMNET = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))


def feats(o):
    return o if torch.is_tensor(o) else o.pooler_output   # transformers>=5 returns an output object


def batch(cs, hw, mean, std):
    a = np.stack([cv2.resize(c, (hw[1], hw[0])) for c in cs]).astype(np.float32) / 255
    return torch.from_numpy((a - mean) / std).permute(0, 3, 1, 2).float().to(DEV)


def main(a):
    os.makedirs(f"{W}/p2", exist_ok=True)
    boxes = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{W}/p1/tracks.csv")):
        boxes[r["id"]].append((int(r["frame"]), *[float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
    # sample boxes per track, then read frames in order once
    want = collections.defaultdict(list)
    for t, obs in boxes.items():
        obs.sort()
        for i in np.unique(np.linspace(0, len(obs) - 1, min(PER_TRACK, len(obs))).astype(int)):
            want[obs[i][0]].append((t, obs[i][1:]))
    crops, keys = [], []
    t0 = time.time()
    for f in sorted(want):
        im = cv2.imread(f"{W}/frames/f_{f:06d}.jpg")
        for t, (x1, y1, x2, y2) in want[f]:
            c = im[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            if c.shape[0] >= 24 and c.shape[1] >= 10:
                crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)); keys.append((t, f))
    print(f"crops {len(crops)} from {len(boxes)} tracks, read in {time.time() - t0:.0f}s", flush=True)

    # --- DINOv2-S per box
    if os.path.exists(f"{W}/p2/emb.npz"):
        print("emb.npz exists, skipping DINOv2", flush=True)
    else:
        embed_dino(crops, keys)
    if not a.no_team: team_zero_shot(a, crops, keys)


def embed_dino(crops, keys):
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(DEV)
    E = []; t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(crops), 128):
            E.append(torch.nn.functional.normalize(m(batch(crops[i:i + 128], (224, 112), *IMNET)), dim=1).cpu().numpy())
    E = np.concatenate(E)
    print(f"dinov2: {len(crops) / (time.time() - t0):.0f} img/s", flush=True)
    np.savez(f"{W}/p2/emb.npz", track=np.array([k[0] for k in keys]), frame=np.array([k[1] for k in keys]), emb=E.astype(np.float32))
    del m; torch.mps.empty_cache()


def team_zero_shot(a, crops, keys):
    # --- SigLIP zero-shot team: torso crops vs colour prompts
    from transformers import AutoProcessor, SiglipModel
    sm = SiglipModel.from_pretrained("google/siglip-base-patch16-224").eval().to(DEV)
    sp = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    labels = {
        "target": f"a photo of a {a.sport} player wearing a {a.color} jersey",
        "other_team": f"a photo of a {a.sport} player wearing a {a.opponent} jersey" if a.opponent else f"a photo of a {a.sport} player wearing a dark jersey",
        "referee": "a photo of a referee in a black and white striped shirt",
        "spectator": "a photo of a person in casual clothes",
    }
    with torch.no_grad():
        tin = sp(text=list(labels.values()), padding="max_length", return_tensors="pt").to(DEV)
        tf = torch.nn.functional.normalize(feats(sm.get_text_features(**tin)), dim=-1)
    by_t = collections.defaultdict(list)
    for (t, _), c in zip(keys, crops): by_t[t].append(c)
    team = {}; t0 = time.time()
    with torch.no_grad():
        tids = list(by_t)
        for i in range(0, len(tids), 32):
            chunk = tids[i:i + 32]; cs, own = [], []
            for t in chunk:
                sel = by_t[t][:: max(1, len(by_t[t]) // TEAM_CROPS)][:TEAM_CROPS]
                cs += [c[: int(0.6 * len(c))] for c in sel]; own += [t] * len(sel)   # upper body
            inp = sp(images=cs, return_tensors="pt").to(DEV)
            imf = torch.nn.functional.normalize(feats(sm.get_image_features(**inp)), dim=-1)
            logits = (imf @ tf.T) * sm.logit_scale.exp() + sm.logit_bias
            pr = torch.softmax(logits, 1).cpu().numpy()
            acc = collections.defaultdict(list)
            for t, p in zip(own, pr): acc[t].append(p)
            for t, ps in acc.items():
                p = np.mean(ps, 0); team[t] = dict(p_target_team=float(p[0]), probs=dict(zip(labels, map(float, p))))
    print(f"siglip team: {len(team)} tracks, {time.time() - t0:.0f}s", flush=True)
    json.dump(team, open(f"{W}/p2/team.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="basketball"); ap.add_argument("--color", default="white")
    ap.add_argument("--opponent", default=""); ap.add_argument("--no_team", action="store_true")
    main(ap.parse_args())
