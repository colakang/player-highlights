"""P4: touch verification / scanning with Qwen2.5-VL-7B (MLX): does the player in the red box have the ball?

Frames are decoded straight from the video (VideoToolbox) and cropped in memory: nothing is written to disk.
  verify (default): detector touch events (touches.json) -> up to --k frames each (event ±--ext s); kept if any says yes
  --scan          : every on-court second with her box (box centre inside the middle --centre of the frame), except
                    seconds within ±--skip s of already verified touches -> runs of "yes" seconds become events
A crop of --scale x her height around her box (drawn red) is resized to 512 px.
Out: {W}/p4/touches_vlm.json (verify) / touches_scan.json (scan), same format as touches.json plus "vlm" answers.

PF_WORK=... python p4/touch_vlm.py [--scan]
"""
import argparse, json, os, subprocess, sys, time
import cv2, numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]
from pfcfg import W, FPS
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="touches.json"); ap.add_argument("--out", default="")
ap.add_argument("--k", type=int, default=3); ap.add_argument("--ext", type=float, default=0.0, help="s added on both sides of an event")
ap.add_argument("--scale", type=float, default=3.0); ap.add_argument("--model", default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
ap.add_argument("--scan", action="store_true"); ap.add_argument("--skip", type=float, default=3.0)
ap.add_argument("--centre", type=float, default=0.8, help="scan: only boxes whose centre is in the middle fraction of the width")
ap.add_argument("--prompt", default="Basketball game. Look at the player inside the red box. Does she have the basketball "
                "in her hands right now (holding, catching, dribbling, passing or shooting it)? Answer only yes or no.")
ap.add_argument("--verified", default="touches_vlm.json", help="scan: verified events whose neighbourhood is skipped")
a = ap.parse_args()
OUT = a.out or ("touches_scan.json" if a.scan else "touches_vlm.json")
HB = {int(round(float(t) * FPS)): b for t, b in json.load(open(f"{W}/p4/her_boxes.json")).items()}
stints = json.load(open(f"{W}/p3/timeline.json"))["intervals"]
model, proc = load(a.model); cfg = load_config(a.model)
PROMPT = apply_chat_template(proc, cfg, a.prompt, num_images=1)
VW, VH = 1920, 1080


def frames(want):
    """yield (f, bgr) for the 6 fps frame numbers in `want` (file f_k = decode index k - 1), streamed in one pass"""
    want = sorted(set(want))
    if not want: return
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{(want[0] - 1) / FPS:.3f}", "-i", os.path.realpath(f"{W}/game.mp4"),
           "-vf", f"fps={FPS:g}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8); n = VW * VH * 3; f = want[0]; ws = set(want)
    while f <= want[-1]:
        raw = p.stdout.read(n)
        if len(raw) < n: break
        if f in ws: yield f, np.frombuffer(raw, np.uint8).reshape(VH, VW, 3)
        f += 1
    p.kill()


def ask(f, im):
    x1, y1, x2, y2 = HB[f]; im = im.copy()
    s = int(min(VH, max(400, a.scale * (y2 - y1)))); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ox = int(min(max(0, cx - s / 2), VW - s)); oy = int(min(max(0, cy - s / 2), VH - s))
    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), max(2, s // 200))
    pil = Image.fromarray(cv2.cvtColor(cv2.resize(im[oy:oy + s, ox:ox + s], (512, 512)), cv2.COLOR_BGR2RGB))
    r = generate(model, proc, PROMPT, [pil], max_tokens=3, verbose=False, temperature=0.0)
    return (getattr(r, "text", str(r)) or "").strip().lower()


t0 = time.time()
if not a.scan:
    EV = json.load(open(f"{W}/p4/{a.src}")); picks = {}
    for i, e in enumerate(EV):
        fs = [f for f in range(int(round((e["t"] - a.ext) * FPS)), int(round((e["t_end"] + a.ext) * FPS)) + 1) if f in HB]
        picks[i] = list(dict.fromkeys(fs[int(j * (len(fs) - 1) / max(1, a.k - 1))] for j in range(min(a.k, len(fs))))) if fs else []
    need = {f for v in picks.values() for f in v}; ans = {}
    for f, im in frames(need):
        ans[f] = ask(f, im)
        if len(ans) % 100 == 0: print(f"  {len(ans)}/{len(need)} frames | {time.time() - t0:.0f} s", flush=True)
    out = [dict(e, vlm=[ans.get(f, "") for f in picks[i]]) for i, e in enumerate(EV) if any(ans.get(f, "").startswith("yes") for f in picks[i])]
    print(f"[touch_vlm] {len(EV)} events, {len(need)} frames -> {len(out)} confirmed | {time.time() - t0:.0f} s")
else:
    V = json.load(open(f"{W}/p4/{a.verified}")) if os.path.exists(f"{W}/p4/{a.verified}") else []
    near = lambda t: any(e["t"] - a.skip <= t <= e["t_end"] + a.skip for e in V)
    lo_x, hi_x = VW * (1 - a.centre) / 2, VW * (1 + a.centre) / 2
    pick = {}
    for f, b in sorted(HB.items()):
        t = f / FPS; s = int(t)
        if not any(s0 <= t <= s1 for s0, s1 in stints) or near(t) or not lo_x <= (b[0] + b[2]) / 2 <= hi_x: continue
        if s not in pick or abs(f / FPS - s - 0.5) < abs(pick[s] / FPS - s - 0.5): pick[s] = f
    yes = []
    for f, im in frames(pick.values()):
        if ask(f, im).startswith("yes"): yes.append(f)
        n_done = n_done + 1 if "n_done" in dir() else 1
        if n_done % 200 == 0: print(f"  {n_done}/{len(pick)} s | yes {len(yes)} | {time.time() - t0:.0f} s", flush=True)
    print(f"  scanned {len(pick)} s -> {len(yes)} yes | {time.time() - t0:.0f} s", flush=True)
    out, cur = [], []
    for f in sorted(yes):
        if cur and f / FPS - cur[-1] / FPS > 2.0: out.append(cur); cur = []
        cur.append(f)
    if cur: out.append(cur)
    out = [dict(t=c[0] / FPS, t_end=c[-1] / FPS, frames=len(c), cues=dict(vlm=len(c)), vlm=["yes"] * len(c)) for c in out]
    print(f"[touch_scan] {len(pick)} seconds scanned -> {len(out)} events | {time.time() - t0:.0f} s")
json.dump(out, open(f"{W}/p4/{OUT}", "w"), indent=0)
