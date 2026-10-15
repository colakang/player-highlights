"""Identity sub-pipeline: where is the target in every frame (for follow-cam, auto highlights, touches).

The main work dir keeps the McByte pipeline (validated for on-court time). This builds {W}_id next to it:
BotSort+OSNet tracks -> number assoc (reuses the main reads) -> tracklet split (jersey colour) -> 7B recheck ->
appearance features -> lineup-constrained assignment inside the main on-court stints.
Exports {W}/p4/her_boxes.json  {"<time_s>": [x1, y1, x2, y2], ...} (6 fps, opponent-colour frames removed).

PF_WORK=<main work dir> python p4/identity.py [--number 12] [--color white]
"""
import argparse, collections, csv, json, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--number", default="12"); ap.add_argument("--color", default="white")
ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--greedy", action="store_true", help="old greedy lineup assignment instead of soft matching"); ap.add_argument("--resplit", action="store_true", help="redo colour split + downstream on existing BotSort tracks")
a = ap.parse_args()
W = os.environ["PF_WORK"].rstrip("/"); I = W + "_id"; PY = sys.executable


def run(*args):
    subprocess.run([PY, "-W", "ignore", *args], cwd=ROOT, env=dict(os.environ, PF_WORK=I), check=True)


for d in ("p1", "p2", "p3", "p4", "audio"): os.makedirs(f"{I}/{d}", exist_ok=True)
link = lambda src, dst: (os.path.lexists(dst) or os.symlink(os.path.abspath(src), dst))
link(f"{W}/frames", f"{I}/frames"); link(f"{W}/p1/det", f"{I}/p1/det"); link(f"{W}/p1/numcrops", f"{I}/p1/numcrops")
if not os.path.lexists(f"{I}/game.mp4"): os.symlink(os.path.realpath(f"{W}/game.mp4"), f"{I}/game.mp4")
for f in ("meta.json", "camera_goals.json", "p1/teamcolor_white.json", "audio/whistles.json"):
    if os.path.exists(f"{W}/{f}") and not os.path.exists(f"{I}/{f}"): shutil.copy(f"{W}/{f}", f"{I}/{f}")
# reads are keyed by crop path: rewrite the prefix so the id dir can reuse them (no re-reading)
if not os.path.exists(f"{I}/p1/reads.jsonl"):
    with open(f"{W}/p1/reads.jsonl") as fi, open(f"{I}/p1/reads.jsonl", "w") as fo:
        for l in fi: fo.write(l.replace(f"{W}/p1/numcrops", f"{I}/p1/numcrops"))
open(f"{I}/p1/read.done", "w").close()
# on-court stints come from the main (McByte) pipeline
shutil.copy(f"{W}/p3/timeline.json", f"{I}/p3/timeline.json")

if a.resplit and os.path.exists(f"{I}/p1/tracks_raw.csv"):
    shutil.copy(f"{I}/p1/tracks_raw.csv", f"{I}/p1/tracks.csv"); shutil.copy(f"{I}/p1/assoc_raw.json", f"{I}/p1/assoc.json")
    for f in ("p1/tracks_raw.csv", "p1/assoc_raw.json", "p3/recheck.json", "p2/emb.npz", "p2/parts.npz"):
        if os.path.exists(f"{I}/{f}"): os.remove(f"{I}/{f}")
    run("p1/split_tracks.py", "--no-embed", "--color", a.color)
if not os.path.exists(f"{I}/p1/tracks_raw.csv"):
    if not os.path.exists(f"{I}/p1/tracks.csv"): run("p1/track_boxmot.py", "--tracker", "botsort")
    run("p1/pipeline.py", "assoc")
    run("p1/split_tracks.py", "--no-embed")
if not os.path.exists(f"{I}/p3/recheck.json"): run("p3/recheck.py", a.number)
if not os.path.exists(f"{I}/p2/emb.npz"): run("p2/embed.py", "--no_team")
if not os.path.exists(f"{I}/p2/parts.npz"): run("p2/parts.py")
run("p4/lineup_id.py", "--number", a.number, "--color", a.color, "--tau", str(a.tau), *([] if a.greedy else ["--soft"]))

# export her boxes, dropping frames where the box wears the opponent colour (two people in one box / leftover switch)
her = set(json.load(open(f"{I}/p4/lineup.json"))["her_tracks"])
col = json.load(open(f"{I}/p1/box_colour.json"))["colour"]
boxes, dropped = {}, 0
for r in csv.DictReader(open(f"{I}/p1/tracks.csv")):
    if r["id"] not in her: continue
    if col.get(r["id"].rsplit("_", 1)[0] + "@" + r["frame"]) == 1: dropped += 1; continue
    boxes[f"{int(r['frame']) / 6.0:.3f}"] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
os.makedirs(f"{W}/p4", exist_ok=True)
json.dump(boxes, open(f"{W}/p4/her_boxes.json", "w"))
on = sum(e - s for s, e in json.load(open(f"{W}/p3/timeline.json"))["intervals"])
secs = len({int(float(t)) for t in boxes})
print(f"[identity] her boxes {len(boxes)} frames ({secs} s = {secs / max(1, on):.0%} of {on} s on court) | opponent-colour frames dropped {dropped}")
