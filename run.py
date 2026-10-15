"""End-to-end: video -> on-court timeline for one player.

python run.py --video clip.mp4 --work work/<game>/q2 --period Q2 --number 12 --color white [--sport basketball]

--period is required: which part of the game the file covers (Q1..Q4, H1/H2, or FULL). Ask the user if unknown.
The source video is read in place (a network share is fine); only derived data goes into --work. Finished steps are skipped.
"""
import argparse, json, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--work", required=True)
ap.add_argument("--period", required=True, help="Q1..Q4 / H1 / H2 / FULL")
ap.add_argument("--number", required=True); ap.add_argument("--color", required=True)
ap.add_argument("--sport", default="basketball", choices=["basketball", "football"])
ap.add_argument("--anchor", nargs="*", default=[], help="mm:ss=on|off (video time), optional user-confirmed states")
ap.add_argument("--gt", default="", help="dev only: on-court GT json for evaluation")
a = ap.parse_args()

W = a.work.rstrip("/"); PY = sys.executable
TEAM = {"basketball": 5, "football": 11}[a.sport]
env = dict(os.environ, PF_WORK=W, PF_GT=a.gt)
os.makedirs(f"{W}/frames", exist_ok=True); os.makedirs(f"{W}/audio", exist_ok=True)
json.dump(vars(a), open(f"{W}/input.json", "w"), indent=1)
if not os.path.lexists(f"{W}/game.mp4"):
    os.symlink(os.path.abspath(a.video), f"{W}/game.mp4")


def step(name, done, cmd):
    if done():
        print(f"[skip] {name}"); return
    t = time.time(); print(f"[run ] {name} ...", flush=True)
    subprocess.run(cmd, env=env, check=True, shell=isinstance(cmd, str))
    print(f"[done] {name} ({time.time() - t:.0f}s)", flush=True)


ex = os.path.exists
step("frames 6fps (VideoToolbox, read in place)", lambda: ex(f"{W}/frames/.done"),
     f'ffmpeg -v error -hwaccel videotoolbox -i "{W}/game.mp4" -vf fps=6 -q:v 3 -start_number 1 "{W}/frames/f_%06d.jpg" && touch "{W}/frames/.done"')
step("audio", lambda: ex(f"{W}/audio/game.wav"),
     f'ffmpeg -v error -y -i "{W}/game.mp4" -vn -ac 1 -ar 22050 "{W}/audio/game.wav"')
step("detect (YOLO11s, CoreML GPU)", lambda: ex(f"{W}/detect.done"),
     f'{PY} -W ignore p1/detect_rf.py > "{W}/detect.log" 2>&1 && grep -q "ALL DONE" "{W}/detect.log" && touch "{W}/detect.done"')
step("whistles", lambda: ex(f"{W}/audio/whistles.json"), [PY, "p3/whistle.py"])
step("track (McByte)", lambda: ex(f"{W}/p1/tracks.csv"), [PY, "-W", "ignore", "p1/pipeline.py", "track"])


def referee_whistles():
    ev = json.load(open(f"{W}/audio/whistles.json"))
    return sum(1 for e in ev if 3300 <= e["f"] <= 4500 and e["loud"] >= -37 and e.get("src") is None)


if not ex(f"{W}/p3/segments.json"):
    dur = json.load(open(f"{W}/meta.json"))["duration"] if ex(f"{W}/meta.json") else \
        float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", f"{W}/game.mp4"]))
    if referee_whistles() < dur / 120:          # this venue's whistle is not usable in the audio (some gyms)
        print(f"[note] only {referee_whistles()} referee whistles found -> vision dead-ball + 10 s grid boundaries", flush=True)
        step("play activity (early, for vision boundaries)", lambda: ex(f"{W}/p3/activity.json"), [PY, "p3/activity.py"])
        subprocess.run([PY, "p3/boundaries_vision.py"], env=env, check=True)
step("segments", lambda: ex(f"{W}/p3/segments.json"), [PY, "p3/segment.py"])
step("number boxes -> players", lambda: ex(f"{W}/p1/assoc.json"), [PY, "-W", "ignore", "p1/pipeline.py", "assoc"])
step("read numbers (Qwen2.5-VL-3B, MLX)", lambda: ex(f"{W}/p1/read.done"),
     f'{PY} -W ignore p1/pipeline.py read > "{W}/read.log" 2>&1 && grep -q "READ DONE" "{W}/read.log" && touch "{W}/p1/read.done"')
step("recheck single reads (Qwen2.5-VL-7B)", lambda: ex(f"{W}/p3/recheck.json"), [PY, "-W", "ignore", "p3/recheck.py", a.number])
step("play activity", lambda: ex(f"{W}/p3/activity.json"), [PY, "p3/activity.py"])
cmd = [PY, "-W", "ignore", "p3/oncourt.py", "--number", a.number, "--color", a.color, "--team_size", str(TEAM)]
if a.anchor: cmd += ["--anchor", *a.anchor]
print("[run ] on-court HMM", flush=True)
out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout
ref = subprocess.run([PY, "-W", "ignore", "p3/refine1s.py", "--number", a.number, "--color", a.color], env=env, check=True,
                     capture_output=True, text=True).stdout                  # coarse-to-fine: edges to 1 s where she is seen
out = "\n".join(l for l in out.splitlines() if not l.startswith("  ")) + "\n" + ref
print("\n".join(l for l in out.splitlines() if not l.startswith("#")))
tl = json.load(open(f"{W}/p3/timeline.json"))
tl.update(period=a.period, number=a.number, color=a.color, sport=a.sport, video=a.video)
json.dump(tl, open(f"{W}/p3/timeline.json", "w"), indent=1)
f = lambda t: f"{int(t) // 60:02d}:{int(t) % 60:02d}"
print(f"\n{a.period} #{a.number} ({a.color}) on court: {sum(e - s for s, e in tl['intervals'])}s")
for s, e in tl["intervals"]:
    print(f"  {a.period} {f(s)} - {f(e)}  ({e - s}s)")
