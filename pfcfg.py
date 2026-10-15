"""Shared per-video config: work dir layout + video duration.

PF_WORK=work/<game>/q1  ->  {W}/game.mp4, {W}/frames/, {W}/audio/, {W}/p1/, {W}/p3/
PF_GT=data/labels/xxx.json  (optional, dev evaluation only)
"""
import json, os, subprocess

W = os.environ.get("PF_WORK", "work")
GT = os.environ.get("PF_GT", "")
FPS = 6.0


def _duration():
    meta = f"{W}/meta.json"
    if os.path.exists(meta):
        return json.load(open(meta))["duration"]
    d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                       "-of", "default=nw=1:nk=1", f"{W}/game.mp4"]).decode().strip())
    json.dump(dict(duration=d), open(meta, "w"))
    return d


DUR = _duration()
