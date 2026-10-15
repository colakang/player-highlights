"""P4: shot vs not-shot classifier for shot candidates (lob passes, referee arms, glare ...).

Features come only from what the pipeline already has (detector boxes at 6 fps, tracks), so it runs on any video.
Trained on user-labelled candidates (data/labels/game_shots.json); evaluated leave-one-quarter-out.
Use: rank candidates; low-score ones go to the END of the review reel (never dropped), so a wrong call costs nothing.

python p4/shot_clf.py eval                      # leave-one-quarter-out report
python p4/shot_clf.py train                     # fit on all labels -> models/shot_clf.pkl
python p4/shot_clf.py score  (with PF_WORK)     # adds "p_shot" to {W}/p4/shot_cands.json
"""
import collections, glob, json, os, pickle, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 6.0
PLAYERS = {"player", "player-in-possession", "player-jump-shot", "player-layup-dunk", "player-shot-block"}
THR = 0.2   # leave-one-quarter-out: p < 0.2 -> 24% of candidates to the end, 3/51 shots among them (both #12 shots >= 0.86)
NAMES = ["n_rise", "n_pose", "n_rim", "ball_frac", "rise_h", "peak_y", "peak_above_heads", "hv_ratio", "air_s",
         "rim_seen", "rim_dist", "toward_rim", "pose_conf", "bib", "n_players_peak", "dur"]


def camera_goals(W):
    """Made-basket times recorded by the Insta360 Flow app (format tag INSFlowBasketballGoalTime, ms). None if absent."""
    cache = f"{W}/camera_goals.json"
    if os.path.exists(cache): return json.load(open(cache))["goals"]
    src = os.path.realpath(f"{W}/game.mp4")
    try:
        tags = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", src]))["format"].get("tags", {})
    except subprocess.CalledProcessError:
        return None                                   # source unreachable (network share offline) -> model without the goal feature
    t = tags.get("INSFlowBasketballGoalTime")
    g = [int(x) / 1000 for x in t.strip("_").split("_") if x] if t is not None else None
    json.dump(dict(goals=g), open(cache, "w"))
    return g


def goal_feat(c, goals):
    return float(any(-1 <= g - c["t"] <= 8 for g in goals))


def load_det(W):
    det = {}
    for f in sorted(glob.glob(f"{W}/p1/det/chunk_*.pkl")): det.update(pickle.load(open(f, "rb")))
    return det


def features(c, det):
    t0, t1 = c["t"], max(c.get("t_end", c["t"]), c["t"])
    fs = [f for f in range(int((t0 - 1.5) * FPS), int((t1 + 2.5) * FPS) + 1) if f in det]
    ball, rims, pose, bib, pl = {}, [], [], 0, {}
    for f in fs:
        d = det[f]; bs = []
        for b, s, k in zip(d["xyxy"], d["conf"], d["cls"]):
            if k == "ball" and s >= 0.3: bs.append((s, b))
            elif k == "rim" and s >= 0.3: rims.append(b)
            elif k in ("player-jump-shot", "player-layup-dunk"): pose.append(float(s))
            elif k == "ball-in-basket": bib += 1
        if bs: ball[f] = max(bs, key=lambda x: x[0])[1]
        pl[f] = [p for p, k, s in zip(d["xyxy"], d["cls"], d["conf"]) if k in PLAYERS and s >= 0.3]
    cues = c.get("cues", {})
    v = dict(n_rise=cues.get("rise", 0), n_pose=cues.get("pose", 0), n_rim=cues.get("rim", 0),
             ball_frac=len(ball) / max(1, len(fs)), rise_h=0.0, peak_y=1.0, peak_above_heads=0.0, hv_ratio=0.0, air_s=0.0,
             rim_seen=float(bool(rims)), rim_dist=3.0, toward_rim=0.0, pose_conf=max(pose, default=0.0), bib=float(bib),
             n_players_peak=0.0, dur=t1 - t0)
    if len(ball) >= 2:
        bf = sorted(ball); cy = {f: (ball[f][1] + ball[f][3]) / 2 for f in bf}; cx = {f: (ball[f][0] + ball[f][2]) / 2 for f in bf}
        peak = min(bf, key=lambda f: cy[f]); low_before = max((cy[f] for f in bf if f <= peak), default=cy[peak])
        ph = np.median([p[3] - p[1] for f in bf for p in pl.get(f, [])] or [200.0])
        v["rise_h"] = (low_before - cy[peak]) / ph
        v["peak_y"] = cy[peak] / 1080.0
        heads = [p[1] for p in pl.get(peak, []) if abs((p[0] + p[2]) / 2 - cx[peak]) < 2 * ph]
        v["peak_above_heads"] = ((min(heads) - cy[peak]) / ph) if heads else 0.0
        first = bf[0]; dx = abs(cx[peak] - cx[first]); dy = abs(cy[first] - cy[peak]) + 1
        v["hv_ratio"] = dx / dy
        v["air_s"] = sum(1 for f in bf if heads and cy[f] < min(heads)) / FPS
        v["n_players_peak"] = float(len(heads))
        if rims:
            rc = np.array([[(r[0] + r[2]) / 2, (r[1] + r[3]) / 2, r[2] - r[0]] for r in rims])
            last = bf[-1]; d = np.hypot(rc[:, 0] - cx[last], rc[:, 1] - cy[last]) / np.maximum(rc[:, 2], 1)
            v["rim_dist"] = float(min(d.min(), 3.0))
            mv = np.array([cx[last] - cx[first], cy[last] - cy[first]]); to = rc[:, :2].mean(0) - np.array([cx[first], cy[first]])
            v["toward_rim"] = float(mv @ to / (np.linalg.norm(mv) * np.linalg.norm(to) + 1e-6))
    return [v[k] for k in NAMES]


LABEL_FILES = sorted(os.path.relpath(f, ROOT) for f in __import__("glob").glob(os.path.join(ROOT, "data/labels/*shots*.json")))   # own reviewed shot labels (not shipped)
POS = {"shot", "ft"}; NEG = {"pass", "none", "other"}      # unlabelled (None) candidates are skipped


def dataset(with_goal=False):
    X, y, grp = [], [], []
    for lf in LABEL_FILES:
        p = os.path.join(ROOT, lf)
        if not os.path.exists(p): continue
        L = json.load(open(p)); game = os.path.basename(lf)
        for q, d in L.items():
            if not q.startswith("Q"): continue
            W = os.path.join(ROOT, d["work"]); det = load_det(W); goals = camera_goals(W) if with_goal else None
            for c in d["candidates"]:
                if c.get("label") not in POS | NEG: continue
                f = features(c, det)
                if with_goal: f.append(goal_feat(c, goals or []))
                X.append(f); y.append(int(c["label"] in POS)); grp.append(f"{game}:{q}")
    return np.array(X, float), np.array(y), np.array(grp)


def model(kind):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, min_samples_leaf=5, class_weight="balanced")


def evaluate():
    X, y, grp = dataset()
    print(f"{len(y)} candidates | shots {y.sum()} | not-shot {len(y) - y.sum()}")
    for kind in ("logreg", "gbt"):
        rows = []
        for q in sorted(set(grp)):
            tr, te = grp != q, grp == q
            m = model(kind).fit(X[tr], y[tr])
            ptr = m.predict_proba(X[tr])[:, 1]; thr = np.min(ptr[y[tr] == 1]) * 0.9     # keep every training shot, with margin
            p = m.predict_proba(X[te])[:, 1]; keep = p >= thr
            rows.append((q, int(te.sum()), int((keep & (y[te] == 1)).sum()), int(y[te].sum()), int((~keep).sum()), int((~keep & (y[te] == 0)).sum())))
        tot = np.array([r[1:] for r in rows]).sum(0)
        print(f"\n[{kind}] leave-one-quarter-out (threshold keeps all training shots x0.9):")
        for q, n, ks, s, dropped, dropped_neg in rows:
            print(f"  {q}: {n} cands | shots kept {ks}/{s} | pushed to end {dropped} (of which not-shot {dropped_neg})")
        print(f"  ALL: shots kept {tot[1]}/{tot[2]} | candidates pushed to end {tot[3]}/{tot[0]} ({tot[3] / tot[0]:.0%}), "
              f"{tot[4]}/{tot[0] - tot[2]} of the not-shots")
        # ranking quality: if the reviewer stops at the first k, how many shots are found
        from sklearn.metrics import roc_auc_score
        p_all = np.zeros(len(y))
        for q in sorted(set(grp)):
            tr, te = grp != q, grp == q; p_all[te] = model(kind).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        print(f"  out-of-quarter AUC {roc_auc_score(y, p_all):.3f}")
    m = model("gbt").fit(X, y)
    try:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(m, X, y, n_repeats=20, random_state=0).importances_mean
        print("\nfeature importance (gbt, permutation):", sorted(zip(NAMES, imp.round(3)), key=lambda x: -x[1])[:8])
    except Exception:
        pass


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "eval":
        evaluate()
    elif cmd == "train":
        M = {}
        for with_goal in (False, True):                       # one model per availability of camera goal tags
            X, y, _ = dataset(with_goal); m = model("logreg").fit(X, y)
            M[with_goal] = dict(model=m, n=len(y))
        os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
        pickle.dump(dict(models=M, thr=THR, features=NAMES), open(os.path.join(ROOT, "models/shot_clf.pkl"), "wb"))
        print(f"trained logreg (with / without camera goal feature) on {M[False]['n']} candidates; push-to-end threshold p < {THR}")
    elif cmd == "score":
        sys.path.insert(0, ROOT); from pfcfg import W
        M = pickle.load(open(os.path.join(ROOT, "models/shot_clf.pkl"), "rb")); det = load_det(W); goals = camera_goals(W)
        m = M["models"][goals is not None]["model"]
        C = json.load(open(f"{W}/p4/shot_cands.json"))
        for c in C:
            f = features(c, det) + ([goal_feat(c, goals)] if goals is not None else [])
            c["p_shot"] = round(float(m.predict_proba(np.array([f]))[0, 1]), 3); c["likely_shot"] = c["p_shot"] >= M["thr"]
        json.dump(C, open(f"{W}/p4/shot_cands.json", "w"), indent=0)
        print(f"scored {len(C)} candidates (camera goal tags: {'yes' if goals is not None else 'no'}) | likely shot {sum(c['likely_shot'] for c in C)} "
              f"| low-priority (to the end of the reel) {sum(not c['likely_shot'] for c in C)}")
