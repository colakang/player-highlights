"""P3 step 2: on-court state per live segment via a 2-state HMM over audio segments.

Evidence per segment = per-track number observations (from P1 reads): target hit / other white number.
Transitions only at segment boundaries, with a prior that depends on the boundary type.

python p3/oncourt.py [--number 12] [--color white] [--anchor 07:40=on ...]
"""
import argparse, collections, csv, json, os, sys
import numpy as np

sys.path.insert(0, "p1")
from pipeline import COLOR_RULES, FPS, DUR, W, GT, team_color_fn

ap = argparse.ArgumentParser()
ap.add_argument("--number", default="12"); ap.add_argument("--color", default="white")
ap.add_argument("--q", type=float, default=0.0, help="P(white-number obs is target | on); 0 = EM from 1/team_size")
ap.add_argument("--team_size", type=int, default=5, help="players on court per team: basketball 5, football 11")
ap.add_argument("--em", type=int, default=10)
ap.add_argument("--verify", type=float, default=0.4, help="appearance score to accept a single-read hit (p3/verify.py)")
ap.add_argument("--recheck_rule", choices=["number", "both", "any", "off"], default="any",
                help="accept a single-read hit if the 7B recheck (p3/recheck.py) reads the target on the number crop / both / either")
ap.add_argument("--fill_gap", type=float, default=120.0, help="hybrid: max gap (s) between Viterbi on-runs that marginals may fill")
ap.add_argument("--many", type=int, default=3, help="target read >= this many times on one track = very strong evidence (0 = off)")
ap.add_argument("--o_cap", type=float, default=-1, help="cap other-number tracks per segment (0 = no cap, -1 = auto: 2 in grid mode)")
ap.add_argument("--edge", type=float, default=0, help="trim each on-interval to [first target read - edge, last read + edge] (0 = off)")
ap.add_argument("--off_cap", type=float, default=0.005, help="max P(confirmed target hit | off court) after EM (0 = no cap)")
ap.add_argument("--thr", type=float, default=0.3, help="P(on) threshold for marginal decoding; lower = recall first")
ap.add_argument("--decode", choices=["marginal", "viterbi", "hybrid"], default="marginal",
                help="viterbi = single most likely state path: a state change is placed at the boundary the evidence supports best")
ap.add_argument("--eps", type=float, default=0.02, help="P(... is the target | off court) — misreads, sideline")
ap.add_argument("--switch", default="whistle=0.1,weak=0.1,buzzer=0.4")
ap.add_argument("--anchor", nargs="*", default=[], help="mm:ss=on|off, user-confirmed states")
ap.add_argument("--dead_act", type=float, default=0.5); ap.add_argument("--dead_min", type=float, default=12.0)
ap.add_argument("--gt", default=GT)
ap.add_argument("--color_mode", choices=["adaptive", "fixed"], default="adaptive")
ap.add_argument("--ambig", choices=["drop", "target", "all"], default="drop",
                help="number boxes inside 2+ overlapping players: drop / keep only target reads (proves presence) / keep all")
a = ap.parse_args()
SW = {k: float(v) for k, v in (x.split("=") for x in a.switch.split(","))}
# grid mode (no usable referee whistle, p3/boundaries_vision.py): 10 s segments -> "nobody read her" is weak evidence
GRID = os.path.exists(f"{W}/audio/whistles.json") and any(e.get("src") == "grid" for e in json.load(open(f"{W}/audio/whistles.json")))
if a.o_cap < 0: a.o_cap = 2 if GRID else 0
if GRID and "--switch" not in sys.argv: SW["weak"] = 0.05

seg = json.load(open(f"{W}/p3/segments.json"))
starts = np.array([s["start"] for s in seg])
seg_of = lambda t: int(np.searchsorted(starts, t, side="right") - 1)

# ---- per-track observations (whole-track read histogram), counted in every segment where the track was read
#      S  = strong hit : target read >= 2x and at least as often as any other number
#      Wk = weak hit   : target read once, nothing else read more often
#      O  = other      : another number dominates (incl. "2"/"20" — confusable teammates are evidence too)
A = json.load(open(f"{W}/p1/assoc.json")); R = {}
for l in open(f"{W}/p1/reads.jsonl"):
    x = json.loads(l); R[x["crop"]] = x["read"].lower().strip(" .")
TEAM = COLOR_RULES[a.color] if a.color_mode == "fixed" else team_color_fn(a.color)
reads = collections.defaultdict(list); segs_of = collections.defaultdict(set)
for x in A:
    rd = R.get(x["crop"])
    if not rd or not rd.isdigit() or not TEAM(*x["bg_hsv"]):
        continue
    if x["ambiguous"] and (a.ambig == "drop" or (a.ambig == "target" and rd != a.number)):
        continue
    reads[x["track"]].append(rd); segs_of[x["track"]].add(seg_of(x["frame"] / FPS))
# weak hits are split by an in-video appearance check (p3/verify.py): Wk+ looks like the strong-hit tracks, Wk- does not
VER = json.load(open(f"{W}/p3/weak_verify.json"))["weak"] if os.path.exists(f"{W}/p3/weak_verify.json") else {}
if a.recheck_rule != "off" and os.path.exists(f"{W}/p3/recheck.json"):       # larger-VLM second opinion supersedes appearance
    RC = json.load(open(f"{W}/p3/recheck.json"))
    ok = lambda v: {"number": v["number_crop_read"] == a.number, "both": v["number_crop_read"] == a.number == v["player_crop_read"],
                    "any": a.number in (v["number_crop_read"], v["player_crop_read"])}[a.recheck_rule]
    VER = {t: (1.0 if ok(v) else 0.0) for t, v in RC.items()}
RCHK = set(RC) if a.recheck_rule != "off" and os.path.exists(f"{W}/p3/recheck.json") else set()
C = np.zeros((len(seg), 5))                                   # columns: S, Wk+, Wk-, O, S+ (many reads)
for t, rs in reads.items():
    c = collections.Counter(rs); n_t = c.pop(a.number, 0); c.pop("1", None)    # lone "1" is too ambiguous
    top = max(c.values(), default=0)
    if n_t >= 2 and n_t >= top and t in RCHK and VER[t] < a.verify: k = 3   # 7B says it is another number (repeated misread)
    elif a.many and n_t >= a.many and n_t >= top: k = 4
    elif n_t >= 2 and n_t >= top: k = 0
    elif n_t == 1 and n_t >= top:
        if t in RCHK: k = 1 if VER[t] >= a.verify else 3       # rejected by the larger VLM = it read another number
        else: k = 1 if VER.get(t, 1.0) >= a.verify else 2
    elif top: k = 3
    else: continue
    for i in segs_of[t]: C[i, k] += 1
if a.o_cap: C[:, 3] = np.minimum(C[:, 3], a.o_cap)
hits, others = C[:, 0] + C[:, 1] + C[:, 4], C[:, 3]                    # for display

# ---- HMM forward-backward (state 0 = off, 1 = on); per-state categorical emissions over (S, Wk, O) learned by EM
T = lambda i: (lambda p: np.array([[1 - p, p], [p, 1 - p]]))(SW.get(seg[i]["boundary"], 0.1))
n = len(seg)
PRIOR = np.array([[0.005, 0.01, 0.04, 0.945, 0.0005],       # off: strong / verified-weak hits are rare, many-read hits ~never
                  [1 / a.team_size * 0.75, 1 / a.team_size * 0.35, 1 / a.team_size * 0.15, 1.0, 1 / a.team_size * 0.5]])
PRIOR = PRIOR / PRIOR.sum(1, keepdims=True); PSEUDO = 20.0


def logE_of(theta):
    logE = C @ np.log(theta).T
    for s_ in a.anchor:
        t, st = s_.split("="); mm, ss = t.split(":"); i = seg_of(int(mm) * 60 + float(ss))
        logE[i] += [0, -50] if st == "off" else [-50, 0]
    return logE


def posterior(theta):
    logE = logE_of(theta); E = np.exp(logE - logE.max(1, keepdims=True))
    fw = np.zeros((n, 2)); bw = np.ones((n, 2))
    fw[0] = 0.5 * E[0]; fw[0] /= fw[0].sum()
    for i in range(1, n):
        fw[i] = (fw[i - 1] @ T(i)) * E[i]; fw[i] /= fw[i].sum()
    for i in range(n - 2, -1, -1):
        bw[i] = T(i + 1) @ (E[i + 1] * bw[i + 1]); bw[i] /= bw[i].sum()
    post = fw * bw
    return post / post.sum(1, keepdims=True)


theta = PRIOR.copy()
for it in range(a.em):
    post = posterior(theta)
    theta = post.T @ C + PSEUDO * PRIOR
    theta = theta / theta.sum(1, keepdims=True)
    if a.off_cap:                                   # EM must not learn 'confirmed #12 reads happen off court' from its own mistakes
        for k in (0, 1, 4): theta[0, k] = min(theta[0, k], a.off_cap)
        theta[0] = theta[0] / theta[0].sum()
post = posterior(theta)
p_on = post[:, 1]
print("emissions (S, Wk+, Wk-, O, S+) | off:", np.round(theta[0], 3), " on:", np.round(theta[1], 3))


def viterbi(theta):
    logE = logE_of(theta)
    V = np.log(0.5) + logE[0]; back = np.zeros((n, 2), int)
    for i in range(1, n):
        cand = V[:, None] + np.log(T(i)); back[i] = cand.argmax(0); V = cand.max(0) + logE[i]
    path = np.zeros(n, int); path[-1] = V.argmax()
    for i in range(n - 1, 0, -1): path[i - 1] = back[i, path[i]]
    return path


if a.decode in ("viterbi", "hybrid"):
    vp = viterbi(theta).astype(bool)
    if a.decode == "hybrid":
        # Viterbi fixes where state changes happen; marginals may only fill gaps *between* on-runs (never extend
        # outward past a Viterbi boundary); a segment with a confirmed target read is always on.
        direct = (C[:, 0] + C[:, 1]) > 0
        on = vp | direct
        idx = np.where(on)[0]
        for i0, i1 in zip(idx, idx[1:]):
            if i1 - i0 > 1 and seg[i1]["start"] - seg[i0]["end"] <= a.fill_gap:
                gap = np.arange(i0 + 1, i1)
                on[gap] |= p_on[gap] >= a.thr
        vp = on
    p_on = np.where(vp, np.maximum(p_on, 0.51), np.minimum(p_on, 0.49))

ts = lambda x: f"{int(x) // 60:02d}:{int(x % 60):02d}"
os.makedirs(f"{W}/p3", exist_ok=True)
json.dump([dict(s, hits=int(hits[i]), others=int(others[i]), p_on=round(float(p_on[i]), 3)) for i, s in enumerate(seg)],
          open(f"{W}/p3/oncourt.json", "w"), indent=0)
for i, s in enumerate(seg):
    bar = "#" * int(round(p_on[i] * 10))
    print(f"#{i:2d} {ts(s['start'])}-{ts(s['end'])} {s['boundary'][:3]:3s} hit {int(hits[i])} oth {int(others[i]):2d}  P(on) {p_on[i]:.2f} {bar}")

# ---- dead time: sustained low play activity (timeouts, breaks, camera off the court)
act = np.array(json.load(open(f"{W}/p3/activity.json"))["moving"])
sm = np.convolve(act, np.ones(9) / 9, "same"); low = sm < a.dead_act
dead = np.zeros(len(low), bool); i = 0
while i < len(low):
    if low[i]:
        j = i
        while j < len(low) and low[j]: j += 1
        if j - i >= a.dead_min: dead[i:j] = True
        i = j
    i += 1

# ---- output intervals: on-court and not dead, at 1 s resolution
sec = np.arange(len(dead))
on_sec = (p_on[np.clip(np.searchsorted(starts, sec + 0.5, side="right") - 1, 0, n - 1)] > (a.thr if a.decode == "marginal" else 0.5)) & ~dead
if a.edge > 0:                                  # trim interval edges to the player's own reads
    rt = sorted(x["frame"] / FPS for x in A if R.get(x["crop"]) == a.number and TEAM(*x["bg_hsv"])
                and (x["track"] not in RCHK or VER.get(x["track"], 0) >= a.verify))
    j = 0
    while j < len(on_sec):
        if on_sec[j]:
            k = j
            while k < len(on_sec) and on_sec[k]: k += 1
            inside = [t for t in rt if j <= t < k]
            if inside:
                on_sec[j:max(j, int(inside[0] - a.edge))] = False
                on_sec[min(k, int(inside[-1] + a.edge) + 1):k] = False
            j = k
        j += 1
iv, i = [], 0
while i < len(on_sec):
    if on_sec[i]:
        j = i
        while j < len(on_sec) and on_sec[j]: j += 1
        iv.append([int(i), int(min(j, DUR))]); i = j
    i += 1
json.dump(dict(intervals=iv, dead=[[int(x) for x in np.where(np.diff(np.r_[0, dead.astype(int), 0]))[0][k:k + 2]] for k in range(0, int(np.abs(np.diff(np.r_[0, dead.astype(int), 0])).sum()), 2)]),
          open(f"{W}/p3/timeline.json", "w"), indent=0)
print(f"on-court intervals: {len(iv)} | total {sum(e - s for s, e in iv)}s | dead time detected {int(dead.sum())}s")
for s_, e_ in iv: print(f"  {ts(s_)} - {ts(e_)}  ({e_ - s_}s)")

# ---- dev eval at 0.1s resolution
if os.path.exists(a.gt):
    g = json.load(open(a.gt))["intervals"]
    grid = np.arange(0, DUR, 0.1)
    gt = np.full(len(grid), "", object)
    for x in g: gt[(grid >= x["start"]) & (grid < x["end"])] = x["state"]
    pred = p_on[np.clip(np.searchsorted(starts, grid, side="right") - 1, 0, n - 1)] > (a.thr if a.decode == "marginal" else 0.5)
    m = np.isin(gt, ["on", "off"])
    tp = (pred & (gt == "on")).sum() / 10; fp = (pred & (gt == "off")).sum() / 10; fn = (~pred & (gt == "on")).sum() / 10
    dead_on = (pred & (gt == "dead")).sum() / 10
    dgrid = dead[np.clip(grid.astype(int), 0, len(dead) - 1)]; fin = pred & ~dgrid
    ftp = (fin & (gt == "on")).sum() / 10; ffp = (fin & np.isin(gt, ["off", "dead"])).sum() / 10
    print(f"[dev] FINAL (on & not dead): recall {ftp / (gt == 'on').sum() * 10:.0%} ({ftp:.0f}/{(gt == 'on').sum() / 10:.0f}s) | "
          f"precision {ftp / max(1e-6, ftp + ffp):.0%} (wrong {ffp:.0f}s: off/dead) | on-court lost to dead filter {(pred & dgrid & (gt == 'on')).sum() / 10:.0f}s")
    print(f"[dev] on-court time: recall {tp / (tp + fn):.0%} ({tp:.0f}/{tp + fn:.0f}s) | precision {tp / max(1e-6, tp + fp):.0%} "
          f"(false on {fp:.0f}s) | predicted-on during dead time {dead_on:.0f}s (handled by dead-time step)")
