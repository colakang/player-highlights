"""P1 step 1: Roboflow basketball detector over all frames of {W}/frames.

Output: {W}/p1/det/chunk_XXXXXX.pkl  {frame_idx: {"xyxy": (N,4), "conf": (N,), "cls": [str]}}
Chunked + resumable. Runs the cached ONNX directly (p1/rfdet.py, CoreML MLProgram on GPU).
"""
import glob, os, pickle, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
from pfcfg import W
from rfdet import YoloOnnx

CONF = 0.15          # keep low; per-class thresholds applied downstream
CHUNK = 256
OUT = f"{W}/p1/det"

os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(f"{W}/frames/f_*.jpg"))
model = YoloOnnx()
t0 = time.time(); done = 0
for c in range(0, len(files), CHUNK):
    path = f"{OUT}/chunk_{c:06d}.pkl"
    if os.path.exists(path):
        continue
    res = {}
    for f in files[c:c + CHUNK]:
        xyxy, conf, cls = model(cv2.imread(f), conf=CONF, iou=0.9)
        res[int(f[-10:-4])] = dict(xyxy=xyxy, conf=conf, cls=cls)
    pickle.dump(res, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
    done += len(res)
    print(f"chunk {c} ok | {done} frames | {done / (time.time() - t0):.1f} fps", flush=True)
print("ALL DONE", flush=True)
