"""Roboflow YOLO11 detector on onnxruntime CoreML (MLProgram -> GPU) + VideoToolbox frame reader.

Replaces `inference.get_model` at runtime: same weights, half the CPU, no dependency pinning.
"""
import glob, os, subprocess
import cv2, numpy as np, onnxruntime as ort

CACHE = "models/rf_cache/models-cache"


def find_weights(model_slug):
    """weights.onnx + class_names.txt: models/detector/ (scripts/setup_models.py) or an `inference` package cache."""
    d = glob.glob("models/detector/weights.onnx") or glob.glob(f"{CACHE}/v2-{model_slug}-*/*/weights.onnx")
    if not d:
        raise FileNotFoundError("detector weights missing: run `python scripts/setup_models.py` (needs ROBOFLOW_API_KEY in .env)")
    root = os.path.dirname(d[0])
    names = [l.strip() for l in open(f"{root}/class_names.txt") if l.strip()]
    return d[0], names


class YoloOnnx:
    def __init__(self, model_slug="basketball-player-detection-3-ycjdo-4", size=1280):
        w, self.names = find_weights(model_slug)
        so = ort.SessionOptions(); so.log_severity_level = 3
        self.s = ort.InferenceSession(w, so, providers=[
            ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}),
            "CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.size = size

    def __call__(self, bgr, conf=0.15, iou=0.9):
        h, w = bgr.shape[:2]
        im = cv2.resize(bgr, (self.size, self.size))       # inference_config: resize_mode = stretch
        x = (im[..., ::-1].transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
        out = self.s.run(None, {self.inp: x})[0][0]           # (4 + nc, anchors)
        scores = out[4:]; cls = scores.argmax(0); sc = scores.max(0)
        keep = sc >= conf
        if not keep.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
        cx, cy, bw, bh = out[:4, keep]; sc, cls = sc[keep], cls[keep]
        sx, sy = w / self.size, h / self.size
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) * np.array([sx, sy, sx, sy])
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w); xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
        kept = []
        for c in np.unique(cls):                               # per-class NMS
            m = np.where(cls == c)[0]
            b = xyxy[m]; i = cv2.dnn.NMSBoxes(np.c_[b[:, :2], b[:, 2:] - b[:, :2]].tolist(), sc[m].tolist(), conf, iou)
            kept += list(m[np.array(i).reshape(-1)])
        kept = np.array(sorted(kept))
        return xyxy[kept].astype(np.float32), sc[kept].astype(np.float32), [self.names[c] for c in cls[kept]]


def video_frames(path, fps=6, start=0.0, dur=None, size=(1920, 1080)):
    """Yield (frame_idx starting at 1, bgr) decoded by VideoToolbox, no JPEG round-trip."""
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", str(start)]
    if dur: cmd += ["-t", str(dur)]
    cmd += ["-i", path, "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    n = size[0] * size[1] * 3; i = 0
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n: break
        i += 1
        yield i, np.frombuffer(buf, np.uint8).reshape(size[1], size[0], 3)
    p.wait()
