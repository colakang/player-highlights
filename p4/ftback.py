"""Free-throw lookback: a free throw is preceded by a stoppage; the play that drew the foul ends at the whistle that
started that stoppage. Used to (a) extend review clips back to the foul, (b) prepend the foul play to #12 FT highlights.

ft_anchor(W, t) -> (is_ft_like, whistle_time | None)
"""
import json
import numpy as np

_cache = {}


def _load(W):
    if W not in _cache:
        act = np.array(json.load(open(f"{W}/p3/activity.json"))["moving"])
        sm = np.convolve(act, np.ones(3) / 3, "same")
        seg = json.load(open(f"{W}/p3/segments.json"))
        ws = sorted(s["start"] for s in seg if s["boundary"] in ("whistle", "weak", "buzzer"))
        _cache[W] = (act, sm, ws)
    return _cache[W]


def ft_like(W, t, thr=1.0):
    act, _, _ = _load(W)
    return float(act[max(0, int(t) - 6):int(t)].mean()) < thr


def ft_anchor(W, t, quiet=1.0, max_back=75.0):
    """walk back over the quiet stretch before t, then take the last whistle at or just after its start."""
    act, sm, ws = _load(W)
    i = int(t) - 1
    while i > 0 and t - i < max_back and (sm[i] < quiet or sm[i - 1] < quiet or sm[max(0, i - 2)] < quiet):
        i -= 1
    # the foul whistle comes before the quiet stretch: players still walk to the line for ~10-20 s after it
    cand = [w for w in ws if i - 25 <= w <= i + 2]
    return (cand[-1] if cand else None)


def ft_groups(W, times, pair_gap=35.0):
    """times of FT-like candidates -> anchor per time; a second FT within pair_gap reuses the first one's anchor."""
    out, last_t, last_a = {}, None, None
    for t in sorted(times):
        a = ft_anchor(W, t)
        if last_t is not None and t - last_t <= pair_gap and last_a is not None:
            a = last_a
        out[t] = a; last_t, last_a = t, a
    return out
