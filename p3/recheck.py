"""Second opinion for single-read target hits with a larger VLM on the number crop AND the player crop.

Out: {W}/p3/recheck.json {track: {"crop": path, "number_crop_read": str, "player_crop_read": str}}
"""
import collections, json, os, sys, time
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pfcfg import W
from verify import track_categories
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

NUMBER = sys.argv[1] if len(sys.argv) > 1 else "12"
MID = sys.argv[2] if len(sys.argv) > 2 else "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
cat = track_categories(NUMBER)
A = json.load(open(f"{W}/p1/assoc.json")); R = {}
for l in open(f"{W}/p1/reads.jsonl"):
    x = json.loads(l); R[x["crop"]] = x["read"].lower().strip(" .")
box = {}
import csv
for r in csv.DictReader(open(f"{W}/p1/tracks.csv")): box[(r["id"], int(r["frame"]))] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
todo = [x for x in A if cat.get(x["track"]) in ("Wk", "S") and R.get(x["crop"]) == NUMBER and not x["ambiguous"]]
# one representative read per track (the largest number box): single reads AND strong tracks (a strong track can be a consistent misread)
best = {}
for x in todo:
    b = x["box"]; area = (b[2] - b[0]) * (b[3] - b[1])
    if x["track"] not in best or area > best[x["track"]][0]: best[x["track"]] = (area, x)
todo = [v for _, v in best.values()]
os.makedirs(f"{W}/p3/recheck", exist_ok=True)
model, proc = load(MID); cfg = load_config(MID)
P_NUM = apply_chat_template(proc, cfg, "This is a crop of a basketball jersey number. Read the number exactly as printed. "
                            "If it is partly hidden, blurred, or not a number, answer 'unclear'. Answer with digits only or 'unclear'.", num_images=1)
P_PLY = apply_chat_template(proc, cfg, "What is the jersey number of this basketball player? If you cannot read it with certainty, "
                            "answer 'unclear'. Answer with digits only or 'unclear'.", num_images=1)
out = {}; t0 = time.time()
for x in todo:
    f = x["frame"]; b = box.get((x["track"], f))
    pc = f"{W}/p3/recheck/p_{x['track']}_{f}.png"
    if b:
        im = cv2.imread(f"{W}/frames/f_{f:06d}.jpg"); x1, y1, x2, y2 = map(int, b)
        c = im[max(0, y1):y2, max(0, x1):x2]; cv2.imwrite(pc, cv2.resize(c, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
    rn = generate(model, proc, P_NUM, [x["crop"]], max_tokens=6, verbose=False, temperature=0.0)
    rp = generate(model, proc, P_PLY, [pc], max_tokens=6, verbose=False, temperature=0.0) if b else None
    g = lambda o: (getattr(o, "text", str(o)) if o is not None else "").strip().lower().strip(" .")
    out[x["track"]] = dict(frame=f, number_crop_read=g(rn), player_crop_read=g(rp))
json.dump(out, open(f"{W}/p3/recheck.json", "w"), indent=0)
print(f"rechecked {len(out)} weak hits in {time.time() - t0:.0f}s")
for t, v in out.items(): print(f"  t{t:>5} number:{v['number_crop_read']:8s} player:{v['player_crop_read']}")
