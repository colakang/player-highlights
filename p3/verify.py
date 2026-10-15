"""P3: verify weak number hits by appearance, learned inside the same video (no labels).

Positives = strong-hit tracks (target read >= 2x and dominant). Negatives = same-colour tracks dominated by another number
(confusable teammates such as #2 / #20 included). A weak-hit track (target read once) is scored by a classifier trained
on those, using track-level appearance (DINOv2 global + 5-stripe colour histograms).

Out: {W}/p3/weak_verify.json  {track: P(target)}
"""
import collections, json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "p1"))
from pfcfg import W
from pipeline import COLOR_RULES, team_color_fn


def track_categories(number="12", color="white"):
    """S / Wk / O per track from the whole-track read histogram (same rule as p3/oncourt.py)."""
    A = json.load(open(f"{W}/p1/assoc.json")); R = {}
    for l in open(f"{W}/p1/reads.jsonl"):
        x = json.loads(l); R[x["crop"]] = x["read"].lower().strip(" .")
    TEAM = team_color_fn(color)
    reads = collections.defaultdict(list)
    for x in A:
        rd = R.get(x["crop"])
        if rd and rd.isdigit() and not x["ambiguous"] and TEAM(*x["bg_hsv"]):
            reads[x["track"]].append(rd)
    cat = {}
    for t, rs in reads.items():
        c = collections.Counter(rs); n_t = c.pop(number, 0); c.pop("1", None); top = max(c.values(), default=0)
        k = "S" if (n_t >= 2 and n_t >= top) else "Wk" if (n_t == 1 and n_t >= top) else "O" if top else None
        if k: cat[t] = k
    return cat


def track_features():
    Z = np.load(f"{W}/p2/emb.npz"); X = np.hstack([Z["emb"], np.load(f"{W}/p2/parts.npz")["X"]])
    X = StandardScaler().fit_transform(X)
    per = collections.defaultdict(list)
    for t, v in zip(Z["track"], X): per[t].append(v)
    return {t: np.mean(v, 0) for t, v in per.items()}


if __name__ == "__main__":
    number = sys.argv[1] if len(sys.argv) > 1 else "12"
    cat = track_categories(number); F = track_features()
    pos = [t for t, k in cat.items() if k == "S" and t in F]; neg = [t for t, k in cat.items() if k == "O" and t in F]
    weak = [t for t, k in cat.items() if k == "Wk" and t in F]
    X = np.array([F[t] for t in pos + neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    pca = PCA(min(32, len(X) - 1), random_state=0).fit(np.array(list(F.values())))
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(pca.transform(X), y)
    sc = {t: float(clf.predict_proba(pca.transform(F[t][None]))[0, 1]) for t in weak}
    # leave-one-out scores of the positives tell us what a true #target track looks like to this model
    loo = []
    for i, t in enumerate(pos):
        m = np.ones(len(y), bool); m[i] = False
        c2 = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(pca.transform(X[m]), y[m])
        loo.append(float(c2.predict_proba(pca.transform(F[t][None]))[0, 1]))
    os.makedirs(f"{W}/p3", exist_ok=True)
    json.dump(dict(weak=sc, pos_loo=dict(zip(pos, loo))), open(f"{W}/p3/weak_verify.json", "w"), indent=0)
    print(f"strong {len(pos)} | other {len(neg)} | weak {len(weak)} | positives LOO score: median {np.median(loo):.2f}, p10 {np.percentile(loo, 10):.2f}")
