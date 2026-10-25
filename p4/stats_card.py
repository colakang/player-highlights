"""Put a game-stats card at the start (default) or end of a video, in place.

python p4/stats_card.py --stats data/stats/<game>.json --video highlight.mp4 [--at start|end] [--sec 3]
stats json: {"title": "...", "subtitle": "...", "minutes": "14:30", "rows": [["投篮", "1/2"], ...], "note": "..."}
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avutil import cjk_font, concat_av

ap = argparse.ArgumentParser()
ap.add_argument("--stats", required=True); ap.add_argument("--video", nargs="+", required=True)
ap.add_argument("--sec", type=float, default=3.0); ap.add_argument("--png", default="")
ap.add_argument("--at", choices=["start", "end"], default="start", help="start: viewers see the stats first")
a = ap.parse_args()
S = json.load(open(a.stats))


def card(path):
    im = Image.new("RGB", (1920, 1080), (14, 22, 38)); d = ImageDraw.Draw(im)
    T, M, B, s = cjk_font(84), cjk_font(48), cjk_font(64), cjk_font(34)
    d.text((140, 110), S["title"], font=T, fill=(255, 255, 255))
    d.text((140, 230), S["subtitle"], font=M, fill=(160, 180, 210))
    d.line((140, 320, 1780, 320), fill=(70, 90, 130), width=3)
    rows = [[S.get("minutes_label", "Minutes"), S["minutes"]]] + S["rows"]
    cols = 2; per = (len(rows) + cols - 1) // cols
    for i, (k, v) in enumerate(rows):
        c, r = divmod(i, per); x = 140 + c * 840; y = 380 + r * 110
        d.text((x, y), k, font=M, fill=(170, 185, 210)); d.text((x + 380, y - 8), v, font=B, fill=(255, 214, 90))
    if S.get("note"): d.text((140, 990), S["note"], font=s, fill=(120, 135, 160))
    im.save(path)


tmp = tempfile.mkdtemp(prefix="card_", dir=os.environ.get("TMPDIR"))
png = a.png or f"{tmp}/card.png"; card(png)
mov = f"{tmp}/card.mov"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", str(a.sec), "-i", png, "-f", "lavfi", "-t", str(a.sec), "-i", "anullsrc=r=44100:cl=stereo",
                "-c:v", "hevc_videotoolbox", "-b:v", "10M", "-tag:v", "hvc1", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "pcm_s16le", mov], check=True)
for v in a.video:
    out = f"{tmp}/out.mp4"; concat_av([mov, v] if a.at == "start" else [v, mov], out, "10M", xfade=0.4)
    shutil.move(out, v); print(f"stats card at the {a.at} ->", v)
shutil.rmtree(tmp, ignore_errors=True)
