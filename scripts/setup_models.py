"""Download the models player-highlights needs (run once; setup.sh calls it).

  detector  : basketball detector ONNX -> models/detector/ (default: this repo's release detector-v2 = YOLO11s trained on
              the CC BY 4.0 Roboflow Universe dataset "basketball-player-detection-3" + hand-corrected youth-game frames;
              re-downloaded when the release tag changes; or --detector roboflow with ROBOFLOW_API_KEY in .env)
  EdgeTAM   : comes with third_party/EdgeTAM (setup.sh clones it)
  Qwen2.5-VL 3B / 7B (MLX 4-bit), DINOv2, OSNet: fetched into the usual caches on first use; --prefetch downloads them now
"""
import argparse, json, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "colakang/player-highlights"
ap = argparse.ArgumentParser()
ap.add_argument("--detector", choices=["release", "roboflow"], default="release")
ap.add_argument("--tag", default="detector-v2"); ap.add_argument("--prefetch", action="store_true")
a = ap.parse_args()
D = os.path.join(ROOT, "models", "detector"); os.makedirs(D, exist_ok=True)


def fetch(url, path):
    print("  downloading", os.path.basename(path), "...", flush=True)
    tmp = path + ".part"; urllib.request.urlretrieve(url, tmp); os.replace(tmp, path)


VER = os.path.join(D, "VERSION")
have = open(VER).read().strip() if os.path.exists(VER) else ("unknown" if os.path.exists(os.path.join(D, "weights.onnx")) else "")
want = a.tag if a.detector == "release" else "roboflow"
if have == want:
    print(f"detector: {have} already present")
elif a.detector == "release":
    base = f"https://github.com/{REPO}/releases/download/{a.tag}"
    fetch(f"{base}/weights.onnx", os.path.join(D, "weights.onnx")); fetch(f"{base}/class_names.txt", os.path.join(D, "class_names.txt"))
    open(VER, "w").write(want + "\n")
else:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env")); key = os.environ.get("ROBOFLOW_API_KEY")
    if not key: sys.exit("set ROBOFLOW_API_KEY in .env (Roboflow Private API Key)")
    r = json.load(urllib.request.urlopen(f"https://api.roboflow.com/ort/basketball-player-detection-3-ycjdo/4?api_key={key}&nocache=true&device=mac&dynamic=true"))["ort"]
    fetch(r["model"], os.path.join(D, "weights.onnx"))
    open(os.path.join(D, "class_names.txt"), "w").write("\n".join(r["classes"]) + "\n")
    open(VER, "w").write(want + "\n")
print("detector:", D)

ck = os.path.join(ROOT, "third_party", "EdgeTAM", "checkpoints", "edgetam.pt")
print("EdgeTAM:", "ok" if os.path.exists(ck) else "missing - run ./setup.sh")
if a.prefetch:
    from huggingface_hub import snapshot_download
    for m in ("mlx-community/Qwen2.5-VL-3B-Instruct-4bit", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"):
        print("  prefetch", m, flush=True); snapshot_download(m)
    import torch
    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
print("done")
