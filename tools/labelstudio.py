"""Fine-tuning step 2: local Label Studio project with the current detector's boxes as pre-labels.

python tools/labelstudio.py start                 # installs Label Studio in .venv_ls (first time), starts it + image server
python tools/labelstudio.py import --token <T>    # creates the project and imports work/labels with pre-labels
python tools/labelstudio.py export --token <T>    # writes the corrected boxes as a YOLO dataset to work/labels/yolo

Label Studio runs on http://localhost:8080 (sign up locally the first time; nothing leaves this Mac).
<T> = Account & Settings -> Personal Access Token (refresh token). Images are served from work/labels on 127.0.0.1:8081.
"""
import argparse, json, os, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "p1")]
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["start", "import", "export"]); ap.add_argument("--token", default="")
ap.add_argument("--dir", default=os.path.join(ROOT, "work", "labels")); ap.add_argument("--project", default="player-highlights labels")
ap.add_argument("--conf", type=float, default=0.3)
a = ap.parse_args()
D = os.path.abspath(a.dir); LS = "http://localhost:8080"
CLASSES = [l.strip() for l in open(os.path.join(ROOT, "models", "detector", "class_names.txt")) if l.strip()]
COLORS = {"number": "#FF3B30", "ball": "#FF9500", "player": "#34C759", "referee": "#8E8E93"}


def api(path, data=None, tok=None, method=None, raw=False):
    h = {"Content-Type": "application/json"}
    if tok: h["Authorization"] = f"Bearer {tok}"
    r = urllib.request.urlopen(urllib.request.Request(LS + path, data=json.dumps(data).encode() if data is not None else None,
                                                      headers=h, method=method), timeout=300)
    return r.read() if raw else json.load(r)


def access():
    if not a.token: sys.exit("--token: Label Studio -> Account & Settings -> Personal Access Token")
    return api("/api/token/refresh", {"refresh": a.token})["access"]


def project(tok):
    ps = api("/api/projects", tok=tok); ps = ps.get("results", ps)
    return next((p["id"] for p in ps if p["title"] == a.project), None)


if a.cmd == "start":
    venv = os.path.join(ROOT, ".venv_ls")
    if not os.path.exists(f"{venv}/bin/label-studio"):
        subprocess.run(["uv", "venv", venv, "--python", "3.12"], check=True)
        subprocess.run(["uv", "pip", "install", "--python", f"{venv}/bin/python", "label-studio"], check=True)
    os.makedirs(D, exist_ok=True)
    subprocess.Popen(f'nohup "{venv}/bin/label-studio" start --no-browser --port 8080 > "{D}/labelstudio.log" 2>&1 &', shell=True)
    subprocess.Popen(f'cd "{D}" && nohup "{sys.executable}" -c "import http.server,functools;'
                     f'H=functools.partial(http.server.SimpleHTTPRequestHandler,directory=\'.\');'
                     f'http.server.ThreadingHTTPServer((\'127.0.0.1\',8081),H).serve_forever()" > /dev/null 2>&1 &', shell=True)
    for _ in range(60):
        try: urllib.request.urlopen(LS, timeout=2); break
        except Exception: time.sleep(2)
    print(f"Label Studio: {LS}  (first time: sign up locally, then copy the Personal Access Token)")

elif a.cmd == "import":
    import cv2
    from rfdet import YoloOnnx
    os.chdir(ROOT); tok = access(); model = YoloOnnx(); tasks = []
    for x in json.load(open(f"{D}/frames.json")):
        im = cv2.imread(f"{D}/images/{x['image']}"); H, W = im.shape[:2]
        xyxy, conf, cls = model(im, conf=a.conf, iou=0.7)
        res = [dict(id=f"p{i}", type="rectanglelabels", from_name="label", to_name="image", original_width=W, original_height=H, score=float(c),
                    value=dict(x=float(b[0]) / W * 100, y=float(b[1]) / H * 100, width=float(b[2] - b[0]) / W * 100,
                               height=float(b[3] - b[1]) / H * 100, rotation=0, rectanglelabels=[k]))
               for i, (b, c, k) in enumerate(zip(xyxy, conf, cls))]
        tasks.append(dict(data=dict(image=f"http://127.0.0.1:8081/images/{x['image']}", **{k: x[k] for k in ("game", "period", "frame", "split")}),
                          predictions=[dict(model_version="detector", result=res)]))
    cfg = '<View><Image name="image" value="$image" zoom="true" zoomControl="true"/><RectangleLabels name="label" toName="image">' + \
          "".join(f'<Label value="{c}" background="{COLORS.get(c, "#5856D6")}"/>' for c in CLASSES) + "</RectangleLabels></View>"
    pid = project(tok) or api("/api/projects", dict(title=a.project, label_config=cfg), tok=tok, method="POST")["id"]
    r = api(f"/api/projects/{pid}/import", tasks, tok=tok, method="POST")
    print(f"project {pid}: imported {r.get('task_count')} frames -> {LS}/projects/{pid}/data")

else:                                                                  # export -> YOLO
    tok = access(); pid = project(tok)
    T = json.loads(api(f"/api/projects/{pid}/export?exportType=JSON", tok=tok, raw=True))
    Y = os.path.join(D, "yolo"); n = {"train": 0, "test": 0}
    for t in T:
        ann = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
        if not ann: continue                                           # not reviewed yet -> not used
        split = t["data"].get("split", "train"); img = os.path.basename(t["data"]["image"].split("?")[0])
        os.makedirs(f"{Y}/{split}/images", exist_ok=True); os.makedirs(f"{Y}/{split}/labels", exist_ok=True)
        src = os.path.join(D, "images", img); dst = f"{Y}/{split}/images/{img}"
        if not os.path.exists(dst):                                    # stretched to 1280x1280 like the base data and inference
            import cv2; cv2.imwrite(dst, cv2.resize(cv2.imread(src), (1280, 1280)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(f"{Y}/{split}/labels/{os.path.splitext(img)[0]}.txt", "w") as fo:
            for r in ann[-1]["result"]:
                v = r.get("value", {}); lab = (v.get("rectanglelabels") or [None])[0]
                if lab not in CLASSES: continue
                cx, cy = (v["x"] + v["width"] / 2) / 100, (v["y"] + v["height"] / 2) / 100
                fo.write(f"{CLASSES.index(lab)} {cx:.6f} {cy:.6f} {v['width'] / 100:.6f} {v['height'] / 100:.6f}\n")
        n[split] += 1
    print(f"exported reviewed frames: train {n['train']}, test {n['test']} -> {Y}")
