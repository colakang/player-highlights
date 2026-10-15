"""P4: trusted identity anchors = frames where her number is actually read.

A read counts when the number crop says --number, is unambiguous, sits on a target-colour background with a
non-opponent jersey, and the 7B recheck of its track (if any) also saw the number.
Anchor box = that track's player box at the frame (identity dir {W}_id from p4/identity.py).
Out: {W}/p4/anchors.json {"<frame>": [x1, y1, x2, y2]}

PF_WORK=<main work dir> python p4/anchors.py [--number 12] [--color white]
"""
import argparse, csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "p1")]
ap = argparse.ArgumentParser()
ap.add_argument("--number", default="12"); ap.add_argument("--color", default="white")
a = ap.parse_args()
W = os.environ["PF_WORK"].rstrip("/"); I = W + "_id"
os.environ["PF_WORK"] = I
import pipeline
team = pipeline.team_color_fn(a.color)
R = {}
for l in open(f"{I}/p1/reads.jsonl"):
    x = json.loads(l); R[x["crop"]] = x["read"].lower().strip(" .")
RC = json.load(open(f"{I}/p3/recheck.json")) if os.path.exists(f"{I}/p3/recheck.json") else {}
ok = lambda t: t not in RC or a.number in (RC[t]["number_crop_read"], RC[t]["player_crop_read"])
want = {}
for x in json.load(open(f"{I}/p1/assoc.json")):
    if R.get(x["crop"]) == a.number and not x["ambiguous"] and x.get("box_colour") != 1 and team(*x["bg_hsv"]) and ok(x["track"]):
        want[(x["track"], x["frame"])] = 1
box = {}
for r in csv.DictReader(open(f"{I}/p1/tracks.csv")):
    k = (r["id"], int(r["frame"]))
    if k in want: box[int(r["frame"])] = [float(r[c]) for c in ("x1", "y1", "x2", "y2")]
json.dump({str(f): b for f, b in sorted(box.items())}, open(f"{W}/p4/anchors.json", "w"))
on = json.load(open(f"{W}/p3/timeline.json"))["intervals"]
fs = sorted(box); ins = [f for f in fs if any(s * 6 <= f <= e * 6 for s, e in on)]
gaps = sorted((b - a_) / 6 for a_, b in zip(ins, ins[1:]))
print(f"[anchors] {len(box)} anchor frames ({len(ins)} on court) | median gap {gaps[len(gaps) // 2] if gaps else 0:.1f} s, "
      f"90% {gaps[int(.9 * len(gaps))] if gaps else 0:.1f} s, max {max(gaps or [0]):.0f} s")
