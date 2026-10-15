"""Lineup-constrained identity assignment of tracklets inside the player's on-court stints.

Within a stint the white players on court are a small set. Tracks with stable number reads label themselves;
the rest of the white tracks are assigned among the lineup by an in-stint appearance classifier, with per-frame
exclusivity (one track per identity per frame). Output: all of the target's tracks/boxes.

PF_WORK=... python p4/lineup_id.py [--number 12] [--tau 0.6]
Out: {W}/p4/lineup.json {"her_tracks": [...], "assign": {track: [number, p]}, "stints": [...]}
"""
import argparse, collections, csv, json, os, sys
import cv2, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "p1"), os.path.join(ROOT, "p3")]
from pfcfg import W, FPS
import pipeline
from verify import track_features

ap = argparse.ArgumentParser()
ap.add_argument("--number", default="12"); ap.add_argument("--color", default="white")
ap.add_argument("--tau", type=float, default=0.6); ap.add_argument("--min_len", type=float, default=1.0)
ap.add_argument("--relay_gap", type=float, default=1.0); ap.add_argument("--relay_dist", type=float, default=0.6)
ap.add_argument("--soft", action="store_true", help="per-frame lineup matching with elimination instead of greedy")
ap.add_argument("--soft_vote", type=float, default=0.5); ap.add_argument("--soft_tau", type=float, default=0.5)
ap.add_argument("--prior_other", type=float, default=0.15); ap.add_argument("--dup_iou", type=float, default=0.6)
ap.add_argument("--edge_min", type=float, default=3.0, help="s; only tracks this long can be dropped as edge-bound")
ap.add_argument("--edge_px", type=float, default=6, help="boxes cut by the left/right frame edge are not identified")
a = ap.parse_args()
FW = 1920
team = pipeline.team_color_fn(a.color)
A = json.load(open(f"{W}/p1/assoc.json")); R = {}
for l in open(f"{W}/p1/reads.jsonl"):
    x = json.loads(l); R[x["crop"]] = x["read"].lower().strip(" .")
RC = json.load(open(f"{W}/p3/recheck.json")) if os.path.exists(f"{W}/p3/recheck.json") else {}
obs = collections.defaultdict(list)
for r in csv.DictReader(open(f"{W}/p1/tracks.csv")):
    obs[r["id"]].append((int(r["frame"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
for v in obs.values(): v.sort()
reads = collections.defaultdict(list); bgw = collections.defaultdict(list)
for x in A:
    rd = R.get(x["crop"])
    if not rd or not rd.isdigit() or x["ambiguous"]: continue
    w = team(*x["bg_hsv"]); bgw[x["track"]].append(w)
    if w: reads[x["track"]].append(rd)


BOXCOL = json.load(open(f"{W}/p1/box_colour.json"))["colour"] if os.path.exists(f"{W}/p1/box_colour.json") else None


def opp_frac(t):
    """share of a tracklet's frames whose jersey is the opponent colour (p1/split_tracks.py); falls back to teal hue"""
    if BOXCOL is None: return teal_frac(t)
    raw = t.rsplit("_", 1)[0]; v = [BOXCOL.get(f"{raw}@{f}", 0) for f, _ in obs[t]]
    k = [c for c in v if c]
    return float(np.mean([c == 1 for c in k])) if k else 0.0


def teal_frac(t):
    o = obs[t]; v = []
    for f, (x1, y1, x2, y2) in [o[len(o) // 4], o[len(o) // 2], o[3 * len(o) // 4]]:
        im = cv2.imread(f"{W}/frames/f_{f:06d}.jpg"); c = im[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        if c.size:
            h = cv2.cvtColor(c[int(.2 * len(c)):int(.55 * len(c))], cv2.COLOR_BGR2HSV)
            v.append(((h[..., 0] > 75) & (h[..., 0] < 105) & (h[..., 1] > 80)).mean())
    return float(np.mean(v)) if v else 1.0


def label(t):
    """stable number of a white track, else None"""
    rs = reads.get(t, [])
    if not rs: return None
    c = collections.Counter(rs); n, k = c.most_common(1)[0]
    if t in RC:                                        # 7B opinion on the target number
        ok = a.number in (RC[t]["number_crop_read"], RC[t]["player_crop_read"])
        if n == a.number and not ok: return None
        if ok and c[a.number] >= 1 and c[a.number] >= k: return a.number
    return n if (k >= 2 and k / len(rs) >= 0.6) else None


def edge(b): return b[0] < a.edge_px or b[2] > FW - a.edge_px


def edge_frac(t):
    """share of a track's boxes cut by the frame edge; short tracks (a player running into view) count as 0"""
    return float(np.mean([edge(b) for _, b in obs[t]])) if len(obs[t]) >= a.edge_min * FPS else 0.0


def box_iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-6)


def soft_assign(un, P, classes, cls, lab, T):
    """per-frame lineup matching with elimination: identities already held by number-labelled tracks in that frame are
    removed, each unlabelled track's probabilities are renormalised over the remaining identities + 'other'
    (bench / referee / duplicate), and visible tracks are matched to identities (Hungarian). A track takes the
    identity it wins in >= --soft_vote of its frames with mean renormalised probability >= --soft_tau."""
    from scipy.optimize import linear_sum_assignment
    Pt = {t: dict(zip(classes, row)) for t, row in zip(un, P)}
    PROBS.update({t: {n: round(float(p), 3) for n, p in d.items()} for t, d in Pt.items()})
    at = collections.defaultdict(list)                                # frame -> [(track, box)] of candidate tracks
    for t in T:
        if t in Pt or lab[t] in cls:
            for f, b in obs[t]:
                if not edge(b): at[f].append((t, b))
    votes = collections.defaultdict(collections.Counter); psum = collections.defaultdict(float); seen = collections.Counter()
    for f, L in at.items():
        held = {lab[t] for t, _ in L if lab[t] in cls}
        U = [(t, b) for t, b in L if t in Pt]
        # duplicates: an unlabelled box overlapping a labelled box (or a longer unlabelled one) is the same person
        lb = [b for t, b in L if lab[t] in cls]
        U = [(t, b) for t, b in U if not any(box_iou(b, q) > a.dup_iou for q in lb)]
        U.sort(key=lambda x: -len(obs[x[0]])); keep = []
        for t, b in U:
            if not any(box_iou(b, q) > a.dup_iou for _, q in keep): keep.append((t, b))
        U = keep
        if not U: continue
        R = [n for n in cls if n not in held]
        C = np.full((len(U), len(R) + len(U)), 0.0); Q = []
        for i, (t, _) in enumerate(U):
            p = np.array([Pt[t].get(n, 0.0) for n in R] + [Pt[t].get("teal", 0.0) + a.prior_other]); p = p / p.sum(); Q.append(p)
            C[i, :len(R)] = -np.log(p[:-1] + 1e-9); C[i, len(R):] = -np.log(p[-1] + 1e-9)
        for i, j in zip(*linear_sum_assignment(C)):
            t = U[i][0]; seen[t] += 1
            if j < len(R): votes[t][R[j]] += 1; psum[(t, R[j])] += Q[i][j]
    for t, v in votes.items():
        n, k = v.most_common(1)[0]
        if k >= a.soft_vote * seen[t] and psum[(t, n)] / k >= a.soft_tau:
            assign[t] = [n, round(float(psum[(t, n)] / k), 3)]


PROBS = {}                                                          # unlabelled track -> class probabilities (soft mode)
F = track_features()
stints = json.load(open(f"{W}/p3/timeline.json"))["intervals"]
assign, her = {}, set()
for s0, s1 in stints:
    T = [t for t, o in obs.items() if sum(s0 <= f / FPS <= s1 for f, _ in o) >= 0.5 * len(o) and len(o) >= a.min_len * FPS and t in F
         and edge_frac(t) <= 0.5]
    lab = {t: label(t) for t in T}
    teal = {t for t in T if lab[t] is None and (bgw[t] and np.mean(bgw[t]) < 0.3 or opp_frac(t) > 0.5)}
    ids = collections.Counter(n for n in lab.values() if n)
    cls = [n for n, k in ids.items() if k >= 2]                    # lineup members with enough labelled tracks
    if a.number not in cls: continue
    Xtr, ytr = [], []
    for t in T:
        if lab[t] in cls: Xtr.append(F[t]); ytr.append(lab[t])
        elif t in teal: Xtr.append(F[t]); ytr.append("teal")
    Xtr = np.array(Xtr); pca = PCA(min(24, len(Xtr) - 1), random_state=0).fit(Xtr)
    clf = LogisticRegression(C=0.3, class_weight="balanced", max_iter=4000).fit(pca.transform(Xtr), ytr)
    un = [t for t in T if lab[t] is None and t not in teal]
    P = clf.predict_proba(pca.transform(np.array([F[t] for t in un]))) if un else np.zeros((0, len(clf.classes_)))
    if a.soft: soft_assign(un, P, list(clf.classes_), cls, lab, T)
    else:
        # per-frame exclusivity: labelled tracks own their identity; unlabelled ones by decreasing probability
        owner = collections.defaultdict(set)
        for t in T:
            if lab[t] in cls:
                for f, _ in obs[t]: owner[(f, lab[t])].add(t)
        cand = sorted(((p, t, n) for t, row in zip(un, P) for n, p in zip(clf.classes_, row) if n != "teal" and p >= a.tau), reverse=True)
        done = set()
        for p, t, n in cand:
            if t in done: continue
            clash = sum(1 for f, _ in obs[t] if owner[(f, n)])
            if clash > 0.2 * len(obs[t]): continue
            assign[t] = [n, round(float(p), 3)]; done.add(t)
            for f, _ in obs[t]: owner[(f, n)].add(t)
    for t in T:
        if lab[t] in cls: assign[t] = [lab[t], 1.0]
    her |= {t for t in T if assign.get(t, [None])[0] == a.number}
    print(f"stint {int(s0)//60:02d}:{int(s0)%60:02d}-{int(s1)//60:02d}:{int(s1)%60:02d}: tracks {len(T)} | labelled {sum(1 for v in lab.values() if v)} "
          f"| teal {len(teal)} | lineup {sorted(cls, key=int)} | unlabelled white {len(un)} -> assigned {sum(1 for t in un if t in assign)}")
# relay: her tracklet ends -> a non-opponent, non-assigned tracklet starting close by within relay_gap takes over
# (and symmetrically backwards). Fills the crowded moments where her number cannot be read.
start = collections.defaultdict(list); end = collections.defaultdict(list)
for t, o in obs.items():
    start[o[0][0]].append(t); end[o[-1][0]].append(t)
cen = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1])
def free(t): return t not in assign and (BOXCOL is None or opp_frac(t) < 0.3) and edge_frac(t) <= 0.5
added, frontier = 0, list(her)
while frontier:
    t = frontier.pop(); o = obs[t]
    for direction in (1, -1):
        f0, b0 = o[-1] if direction == 1 else o[0]; x0, y0, h0 = cen(b0)
        best = None
        for df in range(1, int(a.relay_gap * FPS) + 1):
            for u in (start if direction == 1 else end)[f0 + direction * df]:
                if not free(u): continue
                fu, bu = obs[u][0] if direction == 1 else obs[u][-1]; xu, yu, hu = cen(bu)
                d = ((xu - x0) ** 2 + (yu - y0) ** 2) ** 0.5 / max(h0, 1)
                if d <= a.relay_dist and (best is None or d < best[0]): best = (d, u)
            if best: break
        if best:
            u = best[1]; assign[u] = [a.number, round(1 - best[0], 3)]; her.add(u); frontier.append(u); added += 1
print(f"relay added {added} tracklets")
cov = lambda ts: len({f // int(FPS) for t in ts for f, _ in obs[t]})
lab12 = {t for t in her if assign[t][1] == 1.0}
on = sum(e - s for s, e in stints)
print(f"#{a.number}: number-labelled tracks {len(lab12)} ({cov(lab12)} s) -> + assigned {len(her) - len(lab12)} = {len(her)} tracks ({cov(her)} s) of {on} s on court "
      f"({cov(lab12) / on:.0%} -> {cov(her) / on:.0%})")
os.makedirs(f"{W}/p4", exist_ok=True)
json.dump(dict(her_tracks=sorted(her, key=int), assign=assign, stints=stints, probs=PROBS), open(f"{W}/p4/lineup.json", "w"), indent=0)
