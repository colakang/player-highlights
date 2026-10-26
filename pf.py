"""player-highlights: one command from game videos to a player's highlight reel.

python pf.py --videos Q1.mp4 Q2.mp4 Q3.mp4 Q4.mp4 --periods Q1,Q2,Q3,Q4 --format quarters:7:stop \
             --number 12 --color white --name 12_ALEX --zooms 1,1.6 [--out DIR] [--notify 'cmd "{msg}"']

Videos are read in place (local disk, external drive or network share); results go to --out
(default ~/Movies/player-highlights/<game folder>/), intermediate data to work/ (frame cache removed at the end).

Per video (one period each): on-court time -> identity -> number-anchored mask tracking -> touches (+ Qwen check)
-> shot candidates. Then one auto highlight per zoom level, the full on-court version and a stats card.
Finished steps are skipped on re-run (work dir), so an interrupted run can simply be started again.
"""
import argparse, json, os, re, shlex, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--videos", nargs="+", required=True, help="video files in game order (one per period), or one folder (its videos sorted by name)")
ap.add_argument("--periods", required=True, help="comma list matching --videos: Q1..Q4, H1, H2 or FULL")
ap.add_argument("--format", required=True, help="quarters|halves|full : period minutes : stop|running  (e.g. quarters:7:stop, halves:20:running)")
ap.add_argument("--number", required=True); ap.add_argument("--color", required=True, help="jersey colour of the player's team, e.g. white, black, red, blue")
ap.add_argument("--name", default="", help="label used in captions and file names (default: the number)")
ap.add_argument("--title", default="", help="game title on the stats card")
ap.add_argument("--zooms", default="1", help="comma list of follow-cam zoom levels; 1 = original framing (e.g. 1,1.6)")
ap.add_argument("--sport", default="basketball", choices=["basketball"])
ap.add_argument("--team-size", type=int, default=5, help="players per team on court: 5, or 3 / 4 for small-sided (e.g. half-court 4v4)")
ap.add_argument("--out", default="", help="output folder (default: ~/Movies/player-highlights/<game folder>/)")
ap.add_argument("--work", default="", help="work folder for intermediate data (default: work/<game folder name>)")
ap.add_argument("--no-full", action="store_true", help="skip the full on-court version")
ap.add_argument("--no-reid", action="store_true", help="skip the game-specific re-identification step (~15-20 min per game)")
ap.add_argument("--keep-frames", action="store_true", help="keep the 6 fps frame cache (about 1.5 GB per 15 min of video)")
ap.add_argument("--plays", default="", help="your own play-by-play / clip times (see README): cuts exactly those clips + box score")
ap.add_argument("--auto", action="store_true", help="with --plays: also make the automatic reel")
ap.add_argument("--box", default="", help='stats for the card, any of PTS FG 3P FT REB OREB AST STL BLK TOV PF, e.g. "PTS=8 FG=4/8 FT=0/4 REB=1"')
ap.add_argument("--minutes", default="", help="minutes played shown on the card (e.g. 14:30) instead of the automatic estimate")
ap.add_argument("--lang", choices=["en", "zh"], default="en", help="captions, stats card and report language")
ap.add_argument("--adjust", default="", help="clip edits file (lines copied from the _clips.txt list + '-2 +3' / 'x'), re-renders only")
ap.add_argument("--notify", default="", help='optional shell command run at the end / on error, "{msg}" is replaced by the message')
a = ap.parse_args()

VEXT = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts")
if len(a.videos) == 1 and os.path.isdir(a.videos[0]):
    d = a.videos[0]
    a.videos = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(VEXT) and not f.startswith("."))
    print("videos found:", *[os.path.basename(v) for v in a.videos], sep="\n  ")
missing = [v for v in a.videos if not os.path.isfile(v)]
if missing: sys.exit(f"video not found: {missing[0]}")
periods = [p.strip().upper() for p in a.periods.split(",")]
if len(periods) != len(a.videos): sys.exit("--periods must list one period per video")
kind, plen, clock = (a.format.split(":") + ["", ""])[:3]
if kind not in ("quarters", "halves", "full") or clock not in ("stop", "running"): sys.exit("--format looks like quarters:7:stop or halves:20:running")
plen = float(plen)
name = a.name or a.number
zooms = [float(z) for z in a.zooms.split(",")]
game = os.path.basename(os.path.dirname(os.path.abspath(a.videos[0])))
work = a.work or os.path.join(ROOT, "work", re.sub(r"[^A-Za-z0-9._-]+", "_", game))
out = os.path.abspath(os.path.expanduser(a.out)) if a.out else os.path.join(os.path.expanduser("~/Movies/player-highlights"), game)
os.makedirs(work, exist_ok=True); os.makedirs(out, exist_ok=True)
env0 = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")


def notify(msg):
    print(msg, flush=True)
    if a.notify: subprocess.run(a.notify.replace("{msg}", shlex.quote(msg)), shell=True)


def run(cmd, W=None, log=None):
    env = dict(env0, PF_WORK=W) if W else env0
    t = time.time(); print("  $", " ".join(str(c) for c in cmd[1:4]), "...", flush=True)
    with open(log, "a") if log else open(os.devnull, "w") as fo:
        r = subprocess.run(cmd, env=env, cwd=ROOT, stdout=fo if log else None, stderr=subprocess.STDOUT if log else None)
    if r.returncode:
        notify(f"[player-highlights] {game}: step failed ({' '.join(map(str, cmd[1:3]))}), see {log or 'the console'}"); sys.exit(1)
    print(f"    done ({time.time() - t:.0f}s)", flush=True)


games = []
ANALYSE = (["p4/her_tracks.py"], ["p4/shot_cands.py"], ["p4/shot_clf.py", "score"], ["p4/identity.py", "--number", a.number, "--color", a.color],
           ["p4/anchors.py", "--number", a.number, "--color", a.color], ["p4/mask_anchor.py"])
TOUCH = (["p4/ball_crop.py"], ["p4/touch.py"], ["p4/touch_vlm.py"])
DONE = {"p4/mask_anchor.py": "p4/mask_anchor.json", "p4/touch_vlm.py": "p4/touches_vlm.json"}


def steps(W, log, todo):
    for s in todo:
        if DONE.get(s[0]) and os.path.exists(os.path.join(W, DONE[s[0]])): continue
        run([PY, "-W", "ignore", *s], W=W, log=log)


# 1) per period: on-court time, identity, number-anchored mask tracking
for P, video in zip(periods, a.videos):
    W = os.path.join(work, P.lower()); os.makedirs(W, exist_ok=True); log = os.path.join(W, "pf.log")
    print(f"=== {P}: {video}", flush=True)
    games.append(f"{P}={W}")
    if os.path.exists(os.path.join(W, ".pf_done")) or os.path.exists(os.path.join(W, ".pf_analysed")):
        print("  already analysed"); continue
    run([PY, "run.py", "--video", video, "--work", W, "--period", P, "--number", a.number, "--color", a.color, "--sport", a.sport,
         "--team_size", str(a.team_size)], log=log)
    steps(W, log, ANALYSE)
    open(os.path.join(W, ".pf_analysed"), "w").close()

# 2) whole game: game-specific re-identification learnt from clear number reads corrects / completes her boxes
reid_done = os.path.join(work, ".reid_done")
if not a.no_reid and not os.path.exists(reid_done):
    print("=== game ReID", flush=True); model = os.path.join(work, "reid.pt"); log = os.path.join(work, "reid.log")
    per = [f"{g.split('=')[0].lower()}={g.split('=')[1]}" for g in games]
    run([PY, "-W", "ignore", "p4/reid_game.py", "--games", *per, "--number", a.number, "--epochs", "3", "--near", "1.0", "--out", model], log=log)
    run([PY, "-W", "ignore", "p4/reid_score.py", "--model", model, "--periods", *games], log=log)
    for g in games:
        run([PY, "p4/reid_apply.py"], W=g.split("=")[1], log=log)
        run([PY, "p4/reid_segments.py"], W=g.split("=")[1], log=log)
    open(reid_done, "w").close()

# 3) per period: touches on the final boxes (zoomed ball detection + Qwen check)
for g in games:
    P, W = g.split("=")
    if os.path.exists(os.path.join(W, ".pf_done")): continue
    print(f"=== {P}: touches", flush=True)
    steps(W, os.path.join(W, "pf.log"), TOUCH)
    open(os.path.join(W, ".pf_done"), "w").close()

# minutes played
tot = 0.0; iv = []
ts = lambda t: f"{int(t) // 60:02d}:{int(t) % 60:02d}"
for g in games:
    P, W = g.split("=")
    tl = json.load(open(f"{W}/p3/timeline.json")); dur = json.load(open(f"{W}/meta.json"))["duration"]
    on = sum(e - s for s, e in tl["intervals"]); dead = sum(e - s for s, e in tl.get("dead", []))
    tot += on / 60 if clock == "running" else plen * on / max(1.0, dur - dead)
    iv.append(f"{P} " + ", ".join(f"{ts(s)}-{ts(e)}" for s, e in tl["intervals"]))
mins = f"{int(tot)}:{int(round(tot * 60)) % 60:02d}"
stats = os.path.join(work, "stats_auto.json")
zh = a.lang == "zh"
if a.minutes: mins = a.minutes
BOX_LABELS = {"PTS": ("Points", "得分"), "FG": ("Field goals", "投篮"), "3P": ("3-pointers", "三分"), "FT": ("Free throws", "罚球"),
              "REB": ("Rebounds", "篮板"), "OREB": ("Off. rebounds", "前场篮板"), "AST": ("Assists", "助攻"), "STL": ("Steals", "抢断"),
              "BLK": ("Blocks", "盖帽"), "TOV": ("Turnovers", "失误"), "PF": ("Fouls", "犯规")}
box_rows = []
for kv in a.box.replace(",", " ").split():
    k, _, v = kv.partition("="); k = k.strip().upper()
    if k in BOX_LABELS and v: box_rows.append([BOX_LABELS[k][1 if zh else 0], v.strip()])
json.dump(dict(title=name.replace("_", " "), subtitle=(a.title or game) + ("  ·  自动版" if zh else "  ·  auto"), minutes=mins,
               minutes_label="上场时间" if zh else "Minutes", rows=box_rows or [["片段" if zh else "Clips", "自动识别" if zh else "automatic"]],
               note=("数据由家长提供；片段为自动识别" if zh else "Stats provided by the family; clips found automatically") if box_rows else
                    ("自动版：触球经 Qwen2.5-VL 复核 + 投篮候选" if zh else "Automatic: touches checked by Qwen2.5-VL + shot candidates")),
          open(stats, "w"), ensure_ascii=False)
adjust = ["--adjust", os.path.abspath(a.adjust)] if a.adjust else []
tagz = lambda z: "" if z <= 1 else f"_zoom{z:g}"

made = []
if a.plays:                                                           # the user's own play-by-play -> exactly those clips
    pj, sj, lj = (os.path.join(work, n) for n in ("plays.json", "stats_plays.json", "hl_labels.json"))
    run([PY, "p4/plays.py", os.path.abspath(a.plays), pj, a.number, a.lang])
    run([PY, "p4/game_report.py", "--plays", pj, "--games", *games, "--title", a.title or game, "--name", name.replace("_", " "),
         "--qlen", str(plen), "--clock", clock, "--lang", a.lang, "--out-md", os.path.join(out, f"{name}_summary.md"),
         "--stats-json", sj, "--labels-json", lj])
    for z in zooms:
        f = os.path.join(out, f"{name}_plays{tagz(z)}.mp4")
        run([PY, "-W", "ignore", "p4/cut_highlight.py", "--games", *games, "--number", a.number, "--color", a.color, "--name", name,
             "--mode", "plays", "--labels", lj, "--zoom", str(z if z > 1 else 0), "--out", out, "--outfile", f, *adjust])
        run([PY, "p4/stats_card.py", "--stats", sj, "--video", f]); made.append(f)
    made.append(os.path.join(out, f"{name}_summary.md"))
for z in (zooms if (not a.plays or a.auto) else []):
    f = os.path.join(out, f"{name}_highlight{tagz(z)}.mp4")
    run([PY, "-W", "ignore", "p4/cut_highlight.py", "--games", *games, "--number", a.number, "--color", a.color, "--name", name,
         "--mode", "auto_short", "--touches", "touches_vlm.json", "--zoom", str(z if z > 1 else 0), "--out", out, "--outfile", f, *adjust])
    run([PY, "p4/stats_card.py", "--stats", stats, "--video", f]); made.append(f)
if not a.no_full:
    run([PY, "-W", "ignore", "p4/cut_full.py", "--games", *games, "--name", name, "--out", out])
    for P in periods:
        q = os.path.join(out, f"{name}_full_{P}.mp4")
        if os.path.exists(q): os.remove(q)
    full = os.path.join(out, f"{name}_full_game.mp4")
    run([PY, "p4/stats_card.py", "--stats", stats, "--video", full]); made.append(full)
if not a.keep_frames:
    for g in games:
        subprocess.run(["rm", "-rf", os.path.join(g.split("=")[1], "frames")])
notify(f"[player-highlights] {game} done. Minutes played ~{mins} ({kind}, {clock} clock)\n" + "\n".join(made) +
       "\nOn-court (video time):\n" + "\n".join(iv))
