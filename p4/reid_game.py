"""Game-specific re-identification: learn to tell same-jersey teammates apart from the game's own number reads.

Training data comes for free: tracklets whose jersey number was read clearly (lineup.json assign p = 1.0) give labelled
crops of every player whose number we can read, plus opponents (jersey colour). Crops are decoded straight from the
video (nothing written to disk). DINOv2-S with its last --unfreeze blocks trained + a linear head (cross-entropy).

  test : python p4/reid_game.py --games q1=work/g/q1 q2=... --holdout q4    -> AUC #target vs teammates on the held-out period
  train: python p4/reid_game.py --games ... --out work/g/reid.pt             -> model for p4/reid_apply.py
"""
import argparse, collections, csv, json, os, random, subprocess, sys, time
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument("--games", nargs="+", required=True, help="name=main work dir of one period (its _id dir is used)")
ap.add_argument("--number", default="12"); ap.add_argument("--holdout", default="")
ap.add_argument("--per_track", type=int, default=30); ap.add_argument("--min_track", type=int, default=6)
ap.add_argument("--near", type=float, default=1.0, help="s: only crops this close to a clear number read of the same track (0 = whole track)")
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--unfreeze", type=int, default=4); ap.add_argument("--out", default="")
a = ap.parse_args()
FPS = 6.0; DEV = "mps" if torch.backends.mps.is_available() else "cpu"; H, Wd = 224, 112
random.seed(0); torch.manual_seed(0)


def collect(name, W):
    I = W + "_id"; L = json.load(open(f"{I}/p4/lineup.json")); A = L["assign"]
    col = json.load(open(f"{I}/p1/box_colour.json"))["colour"]
    obs = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{I}/p1/tracks.csv")):
        obs[r["id"]].append((int(r["frame"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
    R = {}
    for l in open(f"{I}/p1/reads.jsonl"):
        x = json.loads(l); R[x["crop"]] = x["read"].lower().strip(" .")
    reads = collections.defaultdict(list)                                 # track -> frames where its own number was read
    for x in json.load(open(f"{I}/p1/assoc.json")):
        if x["track"] in A and R.get(x["crop"]) == A[x["track"]][0] and not x["ambiguous"]: reads[x["track"]].append(x["frame"])
    lab = {}
    for t, o in obs.items():
        if len(o) < a.min_track: continue
        if t in A and A[t][1] == 1.0: lab[t] = A[t][0]                    # clear number read (white team)
        else:
            c = [col.get(t.rsplit("_", 1)[0] + "@" + str(f)) for f, _ in o]
            if sum(x == 1 for x in c) > 0.8 * len(c): lab[t] = "opp"
    items = []
    for t, y in lab.items():
        o = obs[t]
        if a.near and y != "opp":                                         # labels are only trusted near a read (tracks switch people)
            rf = reads.get(t, []); o = [(f, b) for f, b in o if any(abs(f - g) <= a.near * FPS for g in rf)]
            if not o: continue
        k = max(1, len(o) // a.per_track)
        items += [(name, f, b, y, t) for f, b in o[::k][:a.per_track]]
    return items, os.path.realpath(f"{W}/game.mp4")


def crops(items, video):
    """decode only the needed 6 fps frames in one pass; crop each box (with 10 % margin) to 224x112"""
    need = collections.defaultdict(list)
    for i, (_, f, b, _, _) in enumerate(items): need[f].append((i, b))
    fs = sorted(need); out = [None] * len(items)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{(fs[0] - 1) / FPS:.3f}", "-i", video,
                          "-vf", f"fps={FPS:g}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
    f, n, want = fs[0], 1920 * 1080 * 3, set(fs)
    while f <= fs[-1]:
        raw = p.stdout.read(n)
        if len(raw) < n: break
        if f in want:
            im = np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)
            for i, (x1, y1, x2, y2) in need[f]:
                mx, my = 0.1 * (x2 - x1), 0.05 * (y2 - y1)
                c = im[max(0, int(y1 - my)):int(y2 + my), max(0, int(x1 - mx)):int(x2 + mx)]
                if c.size: out[i] = cv2.resize(c, (Wd, H))
        f += 1
    p.kill()
    return out


class Net(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.b = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
        for p_ in self.b.parameters(): p_.requires_grad = False
        for blk in self.b.blocks[-a.unfreeze:]:
            for p_ in blk.parameters(): p_.requires_grad = True
        for p_ in self.b.norm.parameters(): p_.requires_grad = True
        self.head = nn.Linear(384, ncls)

    def emb(self, x): return Fn.normalize(self.b(x), dim=1)
    def forward(self, x): return self.head(self.emb(x) * 16)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def tens(imgs, aug=False):
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255
    if aug:
        if random.random() < 0.5: x = x.flip(3)
        x = x * (0.8 + 0.4 * torch.rand(x.shape[0], 1, 1, 1)) + 0.1 * (torch.rand(x.shape[0], 3, 1, 1) - 0.5)
    return ((x.clamp(0, 1) - MEAN) / STD)


t0 = time.time(); data = []
for spec in a.games:
    name, W = spec.split("="); items, video = collect(name, W); cs = crops(items, video)
    data += [(it, c) for it, c in zip(items, cs) if c is not None]
    print(f"{name}: {len(items)} crops, classes {dict(collections.Counter(it[3] for it in items))}", flush=True)
tr = [d for d in data if d[0][0] != a.holdout]; te = [d for d in data if d[0][0] == a.holdout]
classes = sorted({d[0][3] for d in tr}); ci = {c: i for i, c in enumerate(classes)}
net = Net(len(classes)).to(DEV)
opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad], lr=2e-4, weight_decay=0.05)
cnt = collections.Counter(d[0][3] for d in tr); wts = [1 / cnt[d[0][3]] for d in tr]   # class-balanced sampling


def embed(ds):
    net.eval(); E = []
    with torch.no_grad():
        for i in range(0, len(ds), 64): E.append(net.emb(tens([c for _, c in ds[i:i + 64]]).to(DEV)).cpu())
    return torch.cat(E).numpy()


def heldout_auc():
    """#target vs other white players in the held-out period: similarity to the #target prototype from training crops"""
    from sklearn.metrics import roc_auc_score
    tw = [d for d in te if d[0][3] != "opp"]
    if not tw or not any(d[0][3] == a.number for d in tw): return float("nan")
    proto = embed([d for d in tr if d[0][3] == a.number]).mean(0); proto /= np.linalg.norm(proto)
    s = embed(tw) @ proto
    return roc_auc_score([d[0][3] == a.number for d in tw], s)


if te: print(f"held-out {a.holdout}: AUC before training {heldout_auc():.3f}", flush=True)
for ep in range(a.epochs):
    net.train(); idx = random.choices(range(len(tr)), weights=wts, k=len(tr)); tl = 0
    for i in range(0, len(idx), 48):
        b = [tr[j] for j in idx[i:i + 48]]
        x = tens([c for _, c in b], aug=True).to(DEV); y = torch.tensor([ci[it[3]] for it, _ in b]).to(DEV)
        loss = Fn.cross_entropy(net(x), y, label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(b)
    msg = f"epoch {ep + 1}: loss {tl / len(idx):.3f}"
    if te: msg += f" | held-out {a.holdout} AUC {heldout_auc():.3f}"
    print(msg, f"| {time.time() - t0:.0f} s", flush=True)
if a.out:
    torch.save(dict(state=net.state_dict(), classes=classes, number=a.number, unfreeze=a.unfreeze), a.out); print("saved", a.out)
