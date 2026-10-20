"""Fine-tuning step 3: train the detector further on your labelled frames, compare, and (optionally) install it.

python tools/finetune.py [--epochs 60] [--install]

- base data: the CC BY 4.0 Roboflow Universe set (release dataset-bpd3-v4) is mixed in so the model keeps what it knew
- start: the current detector's PyTorch weights (release detector-v1, or --start <file.pt> e.g. another family's
  shared fine-tune)
- your data: work/labels/yolo (tools/labelstudio.py export); frames marked "test" are only used for the comparison
- result: work/finetune/<run>/weights/best.pt + .onnx and a before/after table on your test frames (mAP50 / mAP50-95,
  ball and number recall). --install copies the ONNX to models/detector/ (the previous one is kept as weights_prev.onnx).

Training uses the Apple GPU (MPS) and ~9 GB memory: about 2-4 hours; close other big apps meanwhile.
Shareable result: the weights + the table (never the images). See README "Improve the detector".
"""
import argparse, os, shutil, subprocess, sys, urllib.request, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "colakang/player-highlights"
ap = argparse.ArgumentParser()
ap.add_argument("--labels", default=os.path.join(ROOT, "work", "labels", "yolo")); ap.add_argument("--start", default="")
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--name", default="ft"); ap.add_argument("--repeat", type=int, default=3, help="weight of your frames vs the public set");
ap.add_argument("--patience", type=int, default=0); ap.add_argument("--install", action="store_true")
a = ap.parse_args()
F = os.path.join(ROOT, "work", "finetune"); os.makedirs(F, exist_ok=True)
try:
    import ultralytics  # noqa: F401
except ImportError:
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, "ultralytics>=8.3"], check=True)
from ultralytics import YOLO


def get(url, path):
    if not os.path.exists(path):
        print("downloading", os.path.basename(path), flush=True); urllib.request.urlretrieve(url, path + ".part"); os.replace(path + ".part", path)
    return path


base = os.path.join(F, "base")
if not os.path.exists(os.path.join(base, "data.yaml")):
    z = get(f"https://github.com/{REPO}/releases/download/dataset-bpd3-v4/bpd3-v4.zip", os.path.join(F, "bpd3-v4.zip"))
    zipfile.ZipFile(z).extractall(base)
start = a.start or get(f"https://github.com/{REPO}/releases/download/detector-v1/detector-v1.pt", os.path.join(F, "detector-v1.pt"))
names = [l.strip() for l in open(os.path.join(ROOT, "models", "detector", "class_names.txt")) if l.strip()]
mine_train, mine_test = os.path.join(a.labels, "train", "images"), os.path.join(a.labels, "test", "images")
if not os.path.isdir(mine_train): sys.exit(f"no labelled training frames in {a.labels} - run tools/labelstudio.py export first")
yaml_train = os.path.join(F, "data.yaml")
open(yaml_train, "w").write("train:\n" + "".join(f"  - {p}\n" for p in [os.path.join(base, "train", "images")] + [mine_train] * a.repeat) +
                            f"val: {os.path.join(base, 'valid', 'images')}\nnc: {len(names)}\nnames: {names}\n")
model = YOLO(start)
model.train(data=yaml_train, imgsz=1280, epochs=a.epochs, batch=a.batch, device="mps", workers=2, patience=a.patience or max(20, a.epochs // 3),
            project=F, name=a.name, exist_ok=True, plots=False, lr0=0.002, warmup_epochs=1)
best = os.path.join(F, a.name, "weights", "best.pt")

if os.path.isdir(mine_test):                                           # before / after on your own test frames
    yt = os.path.join(F, "test.yaml")
    open(yt, "w").write(f"train: {mine_train}\nval: {mine_test}\nnc: {len(names)}\nnames: {names}\n")
    print("\n| model | mAP50 | mAP50-95 | ball recall | number recall |\n|---|---|---|---|---|")
    for label, w in (("before", start), ("after", best)):
        m = YOLO(w).val(data=yt, imgsz=1280, batch=a.batch, device="mps", plots=False, verbose=False)
        r = {names[i]: m.box.r[j] for j, i in enumerate(m.box.ap_class_index)}
        print(f"| {label} | {m.box.map50:.3f} | {m.box.map:.3f} | {r.get('ball', float('nan')):.3f} | {r.get('number', float('nan')):.3f} |")
onnx = YOLO(best).export(format="onnx", imgsz=1280, opset=17)
print("fine-tuned weights:", best, "\nONNX:", onnx)
if a.install:
    d = os.path.join(ROOT, "models", "detector")
    if os.path.exists(os.path.join(d, "weights.onnx")): shutil.copy(os.path.join(d, "weights.onnx"), os.path.join(d, "weights_prev.onnx"))
    shutil.copy(onnx, os.path.join(d, "weights.onnx")); print("installed as models/detector/weights.onnx (previous: weights_prev.onnx)")
