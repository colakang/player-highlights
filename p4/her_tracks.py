"""P4: which tracks are the player, inside her on-court time (needed for 'visible' highlights and shot attribution).

Confirmed  = tracks with strong / rechecked-weak number hits (same rule as p3/oncourt.py).
Candidates = other non-teal tracks overlapping her on-court intervals.
Scored by an in-video appearance classifier (confirmed vs same-team other-number tracks), then temporal exclusivity:
at most one 'her' track per frame (confirmed tracks win).

Out: {W}/p4/her_tracks.json {"confirmed": [...], "appearance": {track: score}, "selected": [...]}
"""
import argparse, collections, csv, json, os, sys
import cv2, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "p1"), os.path.join(ROOT, "p3")]
from pfcfg import W, FPS
from verify import track_categories, track_features

ap = argparse.ArgumentParser()
ap.add_argument("--number", default="12"); ap.add_argument("--color", default="white")
ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--min_len", type=int, default=6)
a = ap.parse_args()

cat = track_categories(a.number, a.color)
RC = json.load(open(f"{W}/p3/recheck.json")) if os.path.exists(f"{W}/p3/recheck.json") else {}
ok = lambda t: a.number in (RC[t]["number_crop_read"], RC[t]["player_crop_read"])
confirmed = {t for t, k in cat.items() if k == "S" or (k == "Wk" and t in RC and ok(t))}
others = {t for t, k in cat.items() if k == "O" or (k == "Wk" and t in RC and not ok(t))}

on = json.load(open(f"{W}/p3/timeline.json"))["intervals"]
on_f = np.zeros(int(max(e for _, e in on) * FPS) + 10, bool) if on else np.zeros(1, bool)
for s, e in on: on_f[int(s * FPS):int(e * FPS) + 1] = True

obs = collections.defaultdict(list)
for r in csv.DictReader(open(f"{W}/p1/tracks.csv")):
    obs[r["id"]].append((int(r["frame"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))


def teal(t):
    o = sorted(obs[t]); v = []
    for f, (x1, y1, x2, y2) in [o[len(o) // 4], o[len(o) // 2], o[3 * len(o) // 4]]:
        im = cv2.imread(f"{W}/frames/f_{f:06d}.jpg"); c = im[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        if c.size:
            h = cv2.cvtColor(c[int(.2 * len(c)):int(.55 * len(c))], cv2.COLOR_BGR2HSV)
            v.append(((h[..., 0] > 75) & (h[..., 0] < 105) & (h[..., 1] > 80)).mean())
    return np.mean(v) if v else 1.0


in_on = {t for t, o in obs.items() if sum(on_f[min(f, len(on_f) - 1)] for f, _ in o) >= 0.5 * len(o)}
cands = [t for t in in_on if t not in confirmed and t not in others and len(obs[t]) >= a.min_len and teal(t) < 0.12]

score = {}
if os.path.exists(f"{W}/p2/emb.npz") and os.path.exists(f"{W}/p2/parts.npz"):      # appearance is optional
    F = track_features()
    pos = [t for t in confirmed if t in F]; neg = [t for t in others if t in F]
    X = np.array([F[t] for t in pos + neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    pca = PCA(min(32, len(X) - 1), random_state=0).fit(np.array(list(F.values())))
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(pca.transform(X), y)
    score = {t: float(clf.predict_proba(pca.transform(F[t][None]))[0, 1]) for t in cands if t in F}

# temporal exclusivity: confirmed first, then appearance candidates by score
taken = collections.defaultdict(list); sel = []
for t in sorted(list(confirmed & in_on) + [t for t, s in score.items() if s >= a.tau], key=lambda t: (t not in confirmed, -score.get(t, 1))):
    fr = [f for f, _ in obs[t]]
    if sum(bool(taken[f]) for f in fr) > 0.2 * len(fr): continue
    sel.append(t)
    for f in fr: taken[f].append(t)
os.makedirs(f"{W}/p4", exist_ok=True)
json.dump(dict(confirmed=sorted(confirmed), appearance=score, selected=sel), open(f"{W}/p4/her_tracks.json", "w"), indent=0)
vis = sum(len(obs[t]) for t in sel) / FPS; onsec = sum(e - s for s, e in on)
print(f"confirmed {len(confirmed)} (in on-court {len(confirmed & in_on)}) | candidates {len(cands)} | selected {len(sel)} "
      f"| her visible {vis:.0f}s of {onsec}s on court ({vis / max(1, onsec):.0%})")
