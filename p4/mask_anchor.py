"""P4 (A'): number-anchored mask tracking -> verified her boxes; identity boxes that conflict are removed.

Seeds are trusted anchors (p4/anchors.py: frames where her number is read). Between consecutive anchors <= --link s
apart EdgeTAM runs forward from the first and backward from the second: frames are 'verified' when the forward pass
lands on the second anchor (IoU >= --land) or both passes agree. Longer gaps get one-way passes of --reach s from each
side ('reach'). Final boxes, per frame:
  verified mask             -> mask box
  reach mask + identity box -> identity box if they agree (IoU >= --keep), else dropped (conflict)
  reach mask only           -> mask box
  identity box only         -> kept (no evidence against it)
  identity box vs verified mask conflict -> dropped
EdgeTAM (facebookresearch/EdgeTAM, cloned by setup.sh into third_party/).

PF_WORK=<work dir> PYTORCH_ENABLE_MPS_FALLBACK=1 python p4/mask_anchor.py
Out: {W}/p4/her_boxes.json (identity-only boxes stay in her_boxes_base.json), {W}/p4/mask_anchor.json
"""
import argparse, json, os, shutil, sys, tempfile, time
import cv2, numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ET = os.path.join(ROOT, "third_party/EdgeTAM")
sys.path[:0] = [ROOT, ET]
from pfcfg import W, FPS
from sam2.build_sam import build_sam2_video_predictor

ap = argparse.ArgumentParser()
ap.add_argument("--link", type=float, default=20.0); ap.add_argument("--reach", type=float, default=15.0)
ap.add_argument("--land", type=float, default=0.4); ap.add_argument("--agree", type=float, default=0.5)
ap.add_argument("--keep", type=float, default=0.3); ap.add_argument("--hmin", type=float, default=0.5)
a = ap.parse_args()

base = f"{W}/p4/her_boxes_base.json"
if not os.path.exists(base): shutil.copy(f"{W}/p4/her_boxes.json", base)
HB = {int(round(float(t) * FPS)): b for t, b in json.load(open(base)).items()}
AN = {int(f): b for f, b in json.load(open(f"{W}/p4/anchors.json")).items()}
stints = json.load(open(f"{W}/p3/timeline.json"))["intervals"]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
pred = build_sam2_video_predictor("configs/edgetam.yaml", f"{ET}/checkpoints/edgetam.pt", device=dev)


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-6)


def mask_box(logits):
    m = (logits[0] > 0).cpu().numpy().astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[k, :4]
    return [float(x), float(y), float(x + w), float(y + h)]


def propagate(f0, f1, seeds):
    """seeds {frame: box} (f0 and/or f1); returns {seed: {frame: box|None}} over f0..f1"""
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    for i, f in enumerate(range(f0, f1 + 1)):
        os.symlink(os.path.abspath(f"{W}/frames/f_{f:06d}.jpg"), f"{tmp}/{i:05d}.jpg")
    out = {}
    with torch.inference_mode():
        st = pred.init_state(video_path=tmp, offload_video_to_cpu=True)
        for f, b in seeds.items():
            pred.reset_state(st)
            pred.add_new_points_or_box(st, frame_idx=f - f0, obj_id=1, box=np.array(b, np.float32))
            out[f] = {f0 + i: mask_box(lg[0]) for i, _, lg in pred.propagate_in_video(st, reverse=(f == f1 and f1 != f0))}
    shutil.rmtree(tmp)
    return out


def sane(b, seed):
    return b is not None and a.hmin <= (b[3] - b[1]) / max(seed[3] - seed[1], 1) <= 1 / a.hmin


NF = sum(n.startswith("f_") for n in os.listdir(f"{W}/frames"))                          # frames are f_000001..f_{NF}
jobs = []
L, R_ = int(a.link * FPS), int(a.reach * FPS)
for s0, s1 in stints:
    lo, hi = max(1, int(s0 * FPS)), min(NF, int(s1 * FPS))
    K = sorted(f for f in AN if lo <= f <= hi)
    if not K: continue
    jobs.append(("bwd", max(lo, K[0] - R_), K[0]))
    for fa, fb in zip(K, K[1:]):
        if fb - fa <= 1: continue
        if fb - fa <= L: jobs.append(("link", fa, fb))
        else: jobs += [("fwd", fa, fa + R_), ("bwd", fb - R_, fb)]
    jobs.append(("fwd", K[-1], min(hi, K[-1] + R_)))
todo = sum(f1 - f0 + 1 for _, f0, f1 in jobs)
print(f"[anchor] {len(AN)} anchors | {len(jobs)} jobs, {todo} frames ({todo / FPS:.0f} s) on {dev}", flush=True)

ver, reach, st, t0, done = {}, {}, dict(link=0, land=0, agree=0, reach=0), time.time(), 0
for j, (kind, f0, f1) in enumerate(jobs):
    if kind == "link":
        r = propagate(f0, f1, {f0: AN[f0], f1: AN[f1]}); F, B = r[f0], r[f1]; st["link"] += 1
        if F.get(f1) is not None and iou(F[f1], AN[f1]) >= a.land:
            st["land"] += 1; ver.update({f: F[f] for f in range(f0 + 1, f1) if sane(F[f], AN[f0])})
        else:
            agree = {f: F[f] for f in range(f0 + 1, f1) if sane(F.get(f), AN[f0]) and B.get(f) and iou(F[f], B[f]) >= a.agree}
            st["agree"] += bool(agree); ver.update(agree)
            # the rest of a failed link is only 'reach' evidence, nearest seed's pass
            for f in range(f0 + 1, f1):
                if f in agree: continue
                b = F.get(f) if f - f0 <= f1 - f else B.get(f)
                if sane(b, AN[f0]): reach.setdefault(f, b)
    else:
        s = f0 if kind == "fwd" else f1
        r = propagate(f0, f1, {s: AN[s]})[s]; st["reach"] += 1
        for f, b in r.items():
            if f != s and sane(b, AN[s]): reach.setdefault(f, b)
    done += f1 - f0 + 1
    if j % 20 == 19:
        el = time.time() - t0
        print(f"  {j + 1}/{len(jobs)} jobs | verified {len(ver)} reach {len(reach)} | {el:.0f} s, eta {el / done * (todo - done):.0f} s", flush=True)

final, c = {}, dict(anchor=0, verified=0, kept=0, conflict_v=0, conflict_r=0, reach_only=0, id_only=0)
for f in sorted(set(HB) | set(ver) | set(reach) | set(AN)):
    hb = HB.get(f)
    if f in AN: final[f] = AN[f]; c["anchor"] += 1
    elif f in ver:
        if hb is not None and iou(hb, ver[f]) < a.keep: c["conflict_v"] += 1
        final[f] = ver[f]; c["verified"] += 1
    elif f in reach:
        if hb is None: final[f] = reach[f]; c["reach_only"] += 1
        elif iou(hb, reach[f]) >= a.keep: final[f] = hb; c["kept"] += 1
        else: c["conflict_r"] += 1
    else: final[f] = hb; c["id_only"] += 1
json.dump({f"{f / FPS:.3f}": b for f, b in final.items()}, open(f"{W}/p4/her_boxes.json", "w"))
json.dump(dict(stats=st, counts=c, verified=sorted(ver), reach=sorted(reach)), open(f"{W}/p4/mask_anchor.json", "w"))
on = sum(e - s for s, e in stints); sec = lambda fs: len({int(f / FPS) for f in fs})
print(f"[anchor] {st} | {c}\n[anchor] her seconds {sec(HB)} -> {sec(final)} of {on:.0f} s on court "
      f"({sec(HB) / on:.0%} -> {sec(final) / on:.0%}) | {time.time() - t0:.0f} s")
