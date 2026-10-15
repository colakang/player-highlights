"""P4: full version — every on-court interval of the player, padded, captioned, concatenated per quarter and for the game.

python p4/cut_full.py --games Q1=work/<game>/q1 Q2=work/<game>/q2 ... --out <output dir> [--pad 2]

Source videos are read in place (local disk or network share). Clips are re-encoded with VideoToolbox HEVC so the caption can be overlaid
(this ffmpeg has no drawtext: captions are PNGs made with PIL and overlaid).
"""
import argparse, shutil, json, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avutil import concat_av, cjk_font

ap = argparse.ArgumentParser()
ap.add_argument("--games", nargs="+", required=True, help="PERIOD=workdir, in game order")
ap.add_argument("--out", required=True); ap.add_argument("--pad", type=float, default=2.0)
ap.add_argument("--name", default="player"); ap.add_argument("--bitrate", default="10M")
a = ap.parse_args()

FONT = cjk_font(54)
os.makedirs(a.out, exist_ok=True)
tmp = tempfile.mkdtemp(prefix="cutfull_", dir=os.environ.get("TMPDIR"))
ts = lambda t: f"{int(t) // 60:02d}:{int(t) % 60:02d}"



def caption(text, path):
    im = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=FONT)
    d.rounded_rectangle((40, 40, 40 + x1 - x0 + 48, 40 + y1 - y0 + 36), 14, fill=(0, 0, 0, 150))
    d.text((40 + 24 - x0, 40 + 18 - y0), text, font=FONT, fill=(255, 255, 255, 255))
    im.save(path)


def merge(iv, pad, dur):
    out = []
    for s, e in sorted(iv):
        s, e = max(0.0, s - pad), min(dur, e + pad)
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out


game_parts = []
for spec in a.games:
    period, W = spec.split("=")
    tl = json.load(open(f"{W}/p3/timeline.json")); dur = json.load(open(f"{W}/meta.json"))["duration"]
    src = os.path.realpath(f"{W}/game.mp4")
    clips = merge(tl["intervals"], a.pad, dur)
    parts = []
    for k, (s, e) in enumerate(clips):
        png = f"{tmp}/{period}_{k}.png"; caption(f"{a.name.replace('_', ' ')}  ·  {period} {ts(s)}", png)
        part = f"{tmp}/{period}_{k:02d}.mov"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-hwaccel", "videotoolbox", "-ss", f"{s:.2f}", "-to", f"{e:.2f}", "-i", src,
                        "-i", png, "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto,format=yuv420p[v];",
                        "-map", "[v]", "-map", "0:a?", "-c:v", "hevc_videotoolbox", "-b:v", a.bitrate, "-tag:v", "hvc1",
                        "-c:a", "pcm_s16le", "-fps_mode", "cfr", "-r", "30", part], check=True)
        parts.append(part)
        print(f"  {period} clip {k + 1}/{len(clips)}  {ts(s)}-{ts(e)}  ({e - s:.0f}s)", flush=True)
    out = f"{a.out}/{a.name}_full_{period}.mp4"
    concat_av(parts, out, a.bitrate)
    print(f"[{period}] {len(clips)} clips, {sum(e - s for s, e in clips):.0f}s -> {out}", flush=True)
    game_parts += parts

out = f"{a.out}/{a.name}_full_game.mp4"
concat_av(game_parts, out, a.bitrate)
print(f"[game] -> {out}")
shutil.rmtree(tmp, ignore_errors=True)
