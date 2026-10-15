"""1-second on-court decoding: Viterbi (= shortest path on the on/off x time graph) at 1 s resolution.

Costs per second t (negative log-likelihood):
  emission: her confirmed track visible at t (only within +-VIS s of its confirmed number reads, to avoid ID-switch frames)
            -> strong 'on'; other-number tracks visible -> weak 'off' (capped); segment-level posterior from p3/oncourt.py as a prior
  transition: switching is cheap at stoppages (whistle / buzzer / sustained low activity) and very expensive during live play
Then the dead-time mask from p3/oncourt.py is applied as before.

PF_WORK=... python p3/refine1s.py [--gt ...]   -> {W}/p3/timeline.json (intervals at 1 s), keeps timeline_seg.json
"""
import argparse, collections, csv, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "p1")]
from pfcfg import W, FPS, GT
import pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--number", default="12"); ap.add_argument("--color", default="white")
ap.add_argument("--vis", type=float, default=10.0, help="seconds around confirmed reads where her track counts as visible")
ap.add_argument("--w_her", type=float, default=3.0); ap.add_argument("--w_other", type=float, default=0.02)
ap.add_argument("--o_cap", type=int, default=3); ap.add_argument("--w_prior", type=float, default=1.0)
ap.add_argument("--p_live", type=float, default=1e-4); ap.add_argument("--p_stop", type=float, default=0.05)
ap.add_argument("--mode", choices=["edges", "full"], default="edges", help="edges = keep segment-level structure, move each transition within +-win s")
ap.add_argument("--win", type=float, default=40.0); ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--snap", type=float, default=20.0, help="move entries back / exits forward to the nearest stoppage within N s (0 = off)")
ap.add_argument("--snap_lo", type=float, default=0.8); ap.add_argument("--snap_exit", type=int, default=1)
ap.add_argument("--gt", default=GT)
a = ap.parse_args()

team = pipeline.team_color_fn(a.color)
seg = json.load(open(f"{W}/p3/oncourt.json"))            # per-segment posterior from the segment-level HMM
act = np.array(json.load(open(f"{W}/p3/activity.json"))["moving"])
N = len(act)
RC = json.load(open(f"{W}/p3/recheck.json")) if os.path.exists(f"{W}/p3/recheck.json") else {}
ok = lambda t: t in RC and a.number in (RC[t]["number_crop_read"], RC[t]["player_crop_read"])
A = json.load(open(f"{W}/p1/assoc.json")); Rd = {}
for l in open(f"{W}/p1/reads.jsonl"):
    x = json.loads(l); Rd[x["crop"]] = x["read"].lower().strip(" .")
reads = collections.defaultdict(list); allr = collections.defaultdict(list)
for x in A:
    r = Rd.get(x["crop"])
    if not r or not r.isdigit() or not team(*x["bg_hsv"]): continue
    allr[x["track"]].append(r)
    if r == a.number: reads[x["track"]].append(x["frame"] / FPS)
her = {t for t, v in reads.items() if ok(t) or (t not in RC and len(v) >= 2)}
other = {t for t, rs in allr.items() if t not in her and collections.Counter(rs).most_common(1)[0][0] != a.number}
seen_her = np.zeros(N); seen_other = np.zeros(N)
for r in csv.DictReader(open(f"{W}/p1/tracks.csv")):
    t, s = r["id"], int(int(r["frame"]) / FPS)
    if s >= N: continue
    if t in her and min(abs(s - q) for q in reads[t]) <= a.vis: seen_her[s] = 1
    elif t in other: seen_other[s] += 1 / FPS
seen_other = np.minimum(np.round(seen_other), a.o_cap)

starts = np.array([s["start"] for s in seg]); pon = np.array([s["p_on"] for s in seg])
prior = np.clip(pon[np.clip(np.searchsorted(starts, np.arange(N) + 0.5, side="right") - 1, 0, len(seg) - 1)], 0.02, 0.98)
# cost[t, state]  (state 0 = off, 1 = on)
cost = np.zeros((N, 2))
cost[:, 0] += a.w_her * seen_her + a.w_prior * -np.log(1 - prior)
cost[:, 1] += a.w_other * seen_other + a.w_prior * -np.log(prior)
# stoppages: referee whistle / buzzer boundaries (+-2 s) or sustained low activity
stop = np.zeros(N, bool)
for s in seg:
    if s["boundary"] in ("whistle", "buzzer", "weak"):
        stop[max(0, int(s["start"]) - 2):int(s["start"]) + 3] = True
sm = np.convolve(act, np.ones(3) / 3, "same"); stop |= sm < 0.6
sw = np.where(stop, -np.log(a.p_stop), -np.log(a.p_live)); stay = np.where(stop, -np.log(1 - a.p_stop), -np.log(1 - a.p_live))
if a.mode == "full":
  D = np.zeros((N, 2)); B = np.zeros((N, 2), int); D[0] = cost[0]
  for t in range(1, N):
    for j in (0, 1):
        c_stay, c_sw = D[t - 1, j] + stay[t], D[t - 1, 1 - j] + sw[t]
        B[t, j] = j if c_stay <= c_sw else 1 - j; D[t, j] = min(c_stay, c_sw) + cost[t, j]
  path = np.zeros(N, int); path[-1] = int(np.argmin(D[-1]))
  for t in range(N - 1, 0, -1): path[t - 1] = B[t, path[t]]

old = json.load(open(f"{W}/p3/timeline.json"))        # fresh segment-level result from p3/oncourt.py (run right before)
if a.mode == "edges":
    # coarse-to-fine: segment-level on/off (without the dead mask) gives the structure; each transition is moved to the
    # second inside +-win that minimises (emission costs of the seconds that change state) + (switch cost at that second)
    base = (prior > a.thr).astype(int)          # same threshold as p3/oncourt.py (recall-first)
    tr = [t for t in range(1, N) if base[t] != base[t - 1]]
    path = base.copy()
    for k, b in enumerate(tr):
        lo = max(tr[k - 1] + 1 if k else 1, int(b - a.win)); hi = min(tr[k + 1] - 1 if k + 1 < len(tr) else N - 1, int(b + a.win))
        before, after = base[b - 1], base[b]                # state left of / right of the transition
        if seen_her[lo:hi + 1].sum() == 0: continue          # no direct evidence near this edge: keep the segment-level position
        best, arg = None, b
        for c in range(lo, hi + 1):                         # new transition at second c: [lo, c) = before, [c, hi] = after
            e = cost[lo:c, before].sum() + cost[c:hi + 1, after].sum() + sw[c]
            if best is None or e < best: best, arg = e, c
        path[lo:arg] = before; path[arg:hi + 1] = after
dead = np.zeros(N, bool)
for s, e in old.get("dead", []): dead[int(s):int(e)] = True
if a.snap > 0:
    # substitutions happen at stoppages: an entry is the start of the last low-activity dip before she is first seen,
    # an exit the end of the first dip after she was last seen (only if such a dip lies within --snap seconds)
    low = sm < a.snap_lo
    trs = [t for t in range(1, N) if path[t] != path[t - 1]]
    for t in trs:
        if path[t] == 1:                                    # off -> on at t
            dips = [u for u in range(max(1, int(t - a.snap)), t + 1) if low[u]]
            if dips:
                u = dips[-1]
                while u > 0 and low[u - 1]: u -= 1
                path[u:t] = 1
        elif a.snap_exit:                                   # on -> off at t
            dips = [u for u in range(t, min(N - 1, int(t + a.snap))) if low[u]]
            if dips:
                u = dips[0]
                while u + 1 < N and low[u + 1]: u += 1
                path[t:u + 1] = 1
on = (path == 1) & ~dead
iv, i = [], 0
while i < N:
    if on[i]:
        j = i
        while j < N and on[j]: j += 1
        iv.append([int(i), int(j)]); i = j
    i += 1
json.dump(old, open(f"{W}/p3/timeline_seg.json", "w"))
json.dump(dict(old, intervals=iv), open(f"{W}/p3/timeline.json", "w"), indent=0)
ts = lambda x: f"{x // 60:02d}:{x % 60:02d}"
print(f"1 s Viterbi: {len(iv)} intervals, {sum(e - s for s, e in iv)} s | her-visible seconds {int(seen_her.sum())}")
for s, e in iv: print(f"  {ts(s)} - {ts(e)}  ({e - s}s)")
if a.gt and os.path.exists(a.gt):
    g = json.load(open(a.gt))["intervals"]; st = np.array([""] * N, object)
    for x in g: st[int(x["start"]):min(N, int(np.ceil(x["end"])))] = x["state"]
    tp = (on & (st == "on")).sum(); fp = (on & np.isin(st, ["off", "dead"])).sum(); fn = (~on & (st == "on")).sum()
    print(f"[dev] 1s: recall {tp / max(1, tp + fn):.0%} ({tp}/{tp + fn}s) | precision {tp / max(1, tp + fp):.0%} (wrong {fp}s)")
