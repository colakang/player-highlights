"""P4: highlight version — only moments where the player is verifiably on screen, plus her labelled plays.

Sources per quarter:
  plays   : user-labelled candidates with a #target role (shot / assist / defence / ...), from data/labels/game_shots.json
  visible : windows around confirmed number reads (strong / 7B-rechecked weak tracks), inside on-court time;
            [first read - pad, last read + pad] within the same track (single read: +-single), clipped to the track span
            -> avoids the parts of a track after an ID switch
Windows are merged (gap < --join) and dropped if shorter than --min.

python p4/cut_highlight.py --games Q1=work/<game>/q1 Q2=work/<game>/q2 ... --out <output dir> [--dry]
"""
import argparse, shutil, collections, csv, json, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "p1"), os.path.join(ROOT, "p4")]
from avutil import concat_av, cjk_font, render_sequence
import codes as CODES
from ftback import ft_groups
import follow

ap = argparse.ArgumentParser()
ap.add_argument("--games", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--number", default="12"); ap.add_argument("--color", default="white"); ap.add_argument("--name", default="player")
ap.add_argument("--pad", type=float, default=2.0); ap.add_argument("--single", type=float, default=3.0)
ap.add_argument("--join", type=float, default=1.0); ap.add_argument("--min", type=float, default=3.0)
ap.add_argument("--labels", default=""); ap.add_argument("--dry", action="store_true")
ap.add_argument("--bitrate", default="10M"); ap.add_argument("--mode", choices=["both", "auto", "auto_short", "plays"], default="both",
                help="auto = only automatically found on-screen windows (no human labels); plays = only labelled plays");
ap.add_argument("--xfade", type=float, default=0.4, help="crossfade between clips in seconds (0 = hard cut)"); ap.add_argument("--suffix", default=""); ap.add_argument("--dump", default=""); ap.add_argument("--bridge", type=float, default=8.0, help="auto: join visibility runs separated by <= N s while on court");
ap.add_argument("--shots", type=int, default=1, help="auto: also add likely-shot candidates during her on-court time (rebounds follow shots)");
ap.add_argument("--tail", type=float, default=3.0, help="auto: seconds kept after she was last identified"); ap.add_argument("--zoom", type=float, default=1.6, help="follow-cam zoom (0 = off)"); ap.add_argument("--ft_back", type=float, nargs=2, default=[8.0, 5.0],
                help="free throws by the player: also show [foul whistle - a, + b]")
ap.add_argument("--pre", type=float, default=4.0, help="auto_short: seconds before a touch (catch / pass release)")
ap.add_argument("--post", type=float, default=5.0, help="auto_short: seconds after a touch (pass arrives, shot reaches the rim)")
ap.add_argument("--touches", default="touches.json", help="auto_short: touch events file in {W}/p4"); ap.add_argument("--merge_gap", type=float, default=3.0, help="clips closer than this are merged"); ap.add_argument("--shot_pre", type=float, default=4.0); ap.add_argument("--shot_post", type=float, default=6.0)
ap.add_argument("--adjust", default="", help="clip edits, one per line: 'Q3 09:46 -2 +3' (start -2 s, end +3 s of the clip "
                "containing 09:46), 'Q3 09:46 x' (drop it), 'Q3 10:20 10:35 +' (add a clip)")
ap.add_argument("--outfile", default="", help="output video path (default: <out>/<name>_highlight_game<suffix>[_follow].mp4)")
a = ap.parse_args()

PLAY_ZH = [("missed shot", "出手未进"), ("made", "出手命中"), ("assist", "助攻"), ("defend", "防守"), ("dribbled", "持球推进"), ("turnover", "接球失误")]
LAB = json.load(open(a.labels)) if a.labels and os.path.exists(a.labels) else {}
FONT = cjk_font(50)
ts = lambda t: f"{int(t) // 60:02d}:{int(t) % 60:02d}"


BOX = {}   # time -> her box, only inside confirmed read windows (avoids post-ID-switch frames)


def windows(period, W):
    os.environ["PF_WORK"] = W
    import importlib, pipeline; importlib.reload(pipeline)
    team = pipeline.team_color_fn(a.color) if os.path.exists(f"{W}/p1/teamcolor_{a.color}.json") or True else None
    on = json.load(open(f"{W}/p3/timeline.json"))["intervals"]; dur = json.load(open(f"{W}/meta.json"))["duration"]
    her = set(json.load(open(f"{W}/p4/her_tracks.json"))["confirmed"])
    A = json.load(open(f"{W}/p1/assoc.json")); R = {}
    for l in open(f"{W}/p1/reads.jsonl"):
        x = json.loads(l); R[x["crop"]] = x["read"].lower().strip(" .")
    span = {}
    for r in csv.DictReader(open(f"{W}/p1/tracks.csv")):
        f = int(r["frame"]); s = span.get(r["id"], (f, f)); span[r["id"]] = (min(s[0], f), max(s[1], f))
    reads = collections.defaultdict(list)
    for x in A:
        if x["track"] in her and R.get(x["crop"]) == a.number and not x["ambiguous"] and team(*x["bg_hsv"]):
            reads[x["track"]].append(x["frame"] / 6.0)
    tb = collections.defaultdict(dict)
    for r in csv.DictReader(open(f"{W}/p1/tracks.csv")):
        if r["id"] in reads: tb[r["id"]][int(r["frame"]) / 6.0] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
    win = []; BOX.clear()
    for t, rs in reads.items():
        lo, hi = span[t][0] / 6.0, span[t][1] / 6.0
        p = a.single if len(rs) == 1 else a.pad
        s, e = max(lo, min(rs) - p), min(hi, max(rs) + p)
        if any(os_ - 1 <= (s + e) / 2 <= oe + 1 for os_, oe in on):
            if a.mode != "plays": win.append([s, e, None])
            BOX.update({k: v for k, v in tb[t].items() if s <= k <= e})
    if a.mode == "auto_short":                    # compressed auto: her touches (+-3 s) + likely shots with her identified nearby
        import bisect
        BOX.clear()
        if os.path.exists(f"{W}/p4/her_boxes.json"): BOX.update({float(k): v for k, v in json.load(open(f"{W}/p4/her_boxes.json")).items()})
        HBt = sorted(BOX); win = []
        for e in (json.load(open(f"{W}/p4/{a.touches}")) if os.path.exists(f"{W}/p4/{a.touches}") else []):
            win.append([max(0, e["t"] - a.pre), min(dur, e["t_end"] + a.post), None])
        for c in (json.load(open(f"{W}/p4/shot_cands.json")) if os.path.exists(f"{W}/p4/shot_cands.json") else []):
            if not c.get("likely_shot", True) or not any(os_ <= c["t"] <= oe for os_, oe in on): continue
            i = bisect.bisect_left(HBt, c["t"] - 10)
            if i < len(HBt) and HBt[i] <= c["t"] + 10:
                win.append([max(0, c["t"] - a.shot_pre), min(dur, max(c.get("t_end", c["t"]), c["t"]) + a.shot_post), None])
        win.sort(); out = []
        for s_, e_, tag in win:
            if out and s_ - out[-1][1] < 1.5: out[-1][1] = max(out[-1][1], e_)
            else: out.append([s_, e_, tag])
        return finish(period, out, dur), dur
    HB = f"{W}/p4/her_boxes.json"
    if os.path.exists(HB):                         # identity sub-pipeline (p4/identity.py): her box in every frame she is seen
        hb = {float(k): v for k, v in json.load(open(HB)).items()}
        BOX.clear(); BOX.update(hb)
        if a.mode != "plays":
            win = [w for w in win if w[2]]          # replace read-based visible windows by continuous visibility runs
            # her identity drops out exactly in crowded action (catch & shoot, rebounds): bridge short gaps while on court
            ts_ = sorted(t for t in hb if any(os_ - 1 <= t <= oe + 1 for os_, oe in on)); run_ = []
            same_stint = lambda t0, t1: any(os_ <= t0 and t1 <= oe for os_, oe in on)
            for t in ts_:
                if run_ and t - run_[-1] > (a.bridge if same_stint(run_[-1], t) else 2.0):
                    if run_[-1] - run_[0] >= 2.0: win.append([max(0, run_[0] - 1.0), min(dur, run_[-1] + a.tail), None])
                    run_ = []
                run_.append(t)
            if run_ and run_[-1] - run_[0] >= 2.0: win.append([max(0, run_[0] - 1.0), min(dur, run_[-1] + a.tail), None])
    if a.mode == "auto" and a.shots and os.path.exists(f"{W}/p4/shot_cands.json"):
        for c in json.load(open(f"{W}/p4/shot_cands.json")):
            if c.get("likely_shot", True) and any(os_ <= c["t"] <= oe for os_, oe in on):
                win.append([max(0, c["t"] - 4.0), min(dur, max(c.get("t_end", c["t"]), c["t"]) + 4.0), None])
    ft_times = []
    for c in (LAB.get(period, {}).get("candidates", []) if a.mode != "auto" else []):
        cs = c.get("codes")
        zh = CODES.caption(cs, a.number) if cs else (c.get("her_zh") or (next((z for k, z in PLAY_ZH if k in c["her"]), None) if c.get("her") else None))
        if not zh: continue
        win.append([max(0, c["t"] - 4.0), min(dur, max(c.get("t_end", c["t"]), c["t"]) + 3.5), zh])
        if "罚球" in zh: ft_times.append(c["t"])
    for x in (LAB.get(period, {}).get("extra", []) if a.mode != "auto" else []):          # plays the candidate finder missed, added from review
        zh = CODES.caption(x["codes"], a.number) if x.get("codes") else x.get("her_zh")
        win.append([max(0, x["t"] - 1.0), min(dur, x["t_end"] + 1.0), zh])
    # free-throw lookback: the play that drew the foul (ends at the whistle that started the stoppage)
    for t, w in ft_groups(W, ft_times).items():
        if w is None: continue
        lo, hi = max(0, w - a.ft_back[0]), min(dur, w + a.ft_back[1])
        hit = [x for x in win if x[2] and x[0] < hi and lo < x[1]]
        if hit:                                   # already labelled (e.g. "12np 12gf"): just make sure the window covers it
            hit[0][0] = min(hit[0][0], lo); hit[0][1] = max(hit[0][1], hi)
        else:
            win.append([lo, hi, "造犯规"])
    win.sort(); out = []
    for s, e, tag in win:
        if out and s - out[-1][1] < a.join:
            out[-1][1] = max(out[-1][1], e)
            if tag and out[-1][2] and tag not in out[-1][2]: out[-1][2] += " → " + tag
            else: out[-1][2] = out[-1][2] or tag
        else: out.append([s, e, tag])
    return finish(period, [w for w in out if w[1] - w[0] >= a.min or w[2]], dur), dur


def clock(x):
    m, s_ = x.split(":"); return int(m) * 60 + float(s_)


ADJ = [l.split("#")[0].split() for l in open(a.adjust)] if a.adjust else []


def finish(period, wins, dur):
    """crossfade compensation (each cut loses xfade at both ends) + the user's clip edits, then re-merge"""
    wins = [[max(0, s - a.xfade), min(dur, e + a.xfade), t] for s, e, t in wins]
    for x in (x for x in ADJ if x and x[0] == period):
        if len(x) >= 4 and x[3] == "+": wins.append([clock(x[1]), clock(x[2]), None]); continue
        t = clock(x[1]); hit = [w for w in wins if w[0] - 1 <= t <= w[1] + 1]
        if not hit: print(f"  [adjust] no clip at {period} {x[1]}"); continue
        w = min(hit, key=lambda w: abs((w[0] + w[1]) / 2 - t))
        if len(x) < 3: continue
        if x[2] == "x": wins.remove(w); continue
        w[0] = max(0, w[0] + float(x[2])); w[1] = min(dur, w[1] + float(x[3] if len(x) > 3 else 0))
    wins.sort(); out = []
    for s, e, t in wins:
        if out and s - out[-1][1] < a.merge_gap:            # a short gap would cut through the action between them
            out[-1][1] = max(out[-1][1], e); out[-1][2] = out[-1][2] or t
        else: out.append([s, e, t])
    return out


def caption(text, path):
    im = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=FONT)
    d.rounded_rectangle((40, 40, 40 + x1 - x0 + 48, 40 + y1 - y0 + 36), 14, fill=(0, 0, 0, 150))
    d.text((64 - x0, 58 - y0), text, font=FONT, fill=(255, 255, 255, 255)); im.save(path)



DUMP = {}
tmp = tempfile.mkdtemp(prefix="hl_", dir=os.environ.get("TMPDIR")); parts = []; total = 0
for spec in a.games:
    period, W = spec.split("=")
    wins, dur = windows(period, W); total += sum(e - s for s, e, _ in wins); DUMP[period] = wins
    print(f"[{period}] {len(wins)} clips, {sum(e - s for s, e, _ in wins):.0f}s | plays: {[ (ts(s), t) for s, e, t in wins if t ]}", flush=True)
    if a.dry: continue
    src = os.path.realpath(f"{W}/game.mp4")
    for k, (s, e, tag) in enumerate(wins):
        png = f"{tmp}/{period}_{k}.png"; caption(f"{a.name.replace('_', ' ')}  ·  {period} {ts(s)}–{ts(e)}" + (f"  ·  {tag}" if tag else ""), png)
        rects = follow.path(BOX, s, e, zoom=a.zoom) if a.zoom > 1 else None
        if rects is None: rects = follow.full_rects(s, e)
        parts.append((src, s, e, png, rects, period, None))
print(f"[game] highlight total {total:.0f}s")
if a.dump: json.dump(DUMP, open(a.dump, "w"))
# clip list next to the video: edit "+0 +0" in place (e.g. "-2 +3") or replace it by "x", then pass the file as --adjust
lst = (os.path.splitext(a.outfile)[0] if a.outfile else f"{a.out}/{a.name}_highlight_game{a.suffix}") + "_clips.txt"
with open(lst if not a.dry else os.devnull, "w") as fo:
    for p, ws in DUMP.items():
        for s, e, t in ws: fo.write(f"{p} {ts(s)} +0 +0    # -> {ts(e)}  {e - s:.0f}s  {t or ''}\n")
if not a.dry:
    out = a.outfile or f"{a.out}/{a.name}_highlight_game{a.suffix}{'_follow' if a.zoom > 1 else ''}.mp4"
    render_sequence([(src, s, e, follow.follow_frames(src, s, e, rects, png, clock=p), len(rects)) for src, s, e, png, rects, p, _ in parts],
                    out, a.bitrate, xfade=a.xfade); print("->", out)
shutil.rmtree(tmp, ignore_errors=True)
