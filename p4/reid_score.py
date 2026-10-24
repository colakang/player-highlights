"""ReID scores for every tracked (non-opponent) box on court -> cache for rule search / application.

python p4/reid_score.py --model work/reid_<game>.pt --periods Q1=work/<game>/q1 ...
Out: {W}/p4/reid_scores.json {"<frame>": [[x1, y1, x2, y2, score, track], ...]}   (score = cosine to the #target prototype)
"""
import argparse, collections, csv, json, os, subprocess, sys
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--periods", nargs="+", required=True)
E = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["reid_game.py", "--games", *[p.split("=", 1)[0].lower() + "=" + p.split("=", 1)[1] for p in E.periods], "--epochs", "0"]
exec(open(os.path.join(ROOT, "p4", "reid_game.py")).read().split("t0 = time.time(); data = []")[0])
ck = torch.load(E.model, map_location="cpu"); net = Net(len(ck["classes"])); net.load_state_dict(ck["state"]); net = net.to(DEV).eval()


def emb(imgs):
    with torch.no_grad(): return torch.cat([net.emb(tens(imgs[i:i + 128]).to(DEV)).cpu() for i in range(0, len(imgs), 128)]).numpy()


P = []
for spec in E.periods:
    name, W = spec.split("="); items, video = collect(name, W)
    sel = [it for it in items if it[3] == ck["number"]][::2]; cs = [c for c in crops(sel, video) if c is not None]
    if cs: P.append(emb(cs))
proto = np.vstack(P).mean(0); proto /= np.linalg.norm(proto)
for spec in E.periods:
    name, W = spec.split("="); I = W + "_id"
    on = json.load(open(f"{W}/p3/timeline.json"))["intervals"]; col = json.load(open(f"{I}/p1/box_colour.json"))["colour"]
    items = []
    for r in csv.DictReader(open(f"{I}/p1/tracks.csv")):
        f = int(r["frame"])
        if not any(s * 6 - 6 <= f <= e * 6 + 6 for s, e in on): continue
        if col.get(r["id"].rsplit("_", 1)[0] + "@" + r["frame"]) == 1: continue
        items.append((name, f, [float(r[k]) for k in ("x1", "y1", "x2", "y2")], "?", r["id"]))
    out = collections.defaultdict(list)
    for i in range(0, len(items), 6000):                                   # chunks keep memory bounded
        chunk = items[i:i + 6000]; cs = crops(chunk, os.path.realpath(f"{W}/game.mp4"))
        ok = [(it, c) for it, c in zip(chunk, cs) if c is not None]
        S = emb([c for _, c in ok]) @ proto
        for (it, _), s in zip(ok, S): out[it[1]].append([*map(lambda v: round(v, 1), it[2]), round(float(s), 4), it[4]])
    json.dump(out, open(f"{W}/p4/reid_scores.json", "w"))
    print(f"[reid_score] {name}: {sum(len(v) for v in out.values())} boxes in {len(out)} frames", flush=True)
