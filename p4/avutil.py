"""Shared A/V helpers for the cutters."""
import subprocess


def _dur(p):
    import json
    j = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", p]))
    return min(float(s["duration"]) for s in j["streams"] if s["codec_type"] in ("video", "audio"))


def concat_av(parts, out, vbitrate, xfade=0.0):
    """concat *filter*: each segment starts with audio and video re-aligned (shorter audio is padded with silence),
    so per-part length mismatches (VFR source vs 30 fps CFR parts) cannot accumulate into A/V drift.
    (The concat demuxer with stream copy drifted up to -0.5 s over 22 parts.)
    xfade > 0: crossfade (video xfade + audio acrossfade) of that many seconds between consecutive parts."""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for p in parts: cmd += ["-hwaccel", "videotoolbox", "-i", p]       # hw decode of every part (sw decode cost ~176% CPU)
    n = len(parts)
    if xfade <= 0 or n < 2:
        fc = "".join(f"[{i}:v][{i}:a]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]"
    else:
        d = [_dur(p) for p in parts]
        xf = min(xfade, 0.45 * min(d))
        norm = "".join(f"[{i}:v]fps=30,format=yuv420p,setsar=1,settb=1/30,setpts=PTS-STARTPTS[v{i}];"
                       f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{d[i]:.3f},asetpts=PTS-STARTPTS[a{i}];" for i in range(n))
        chain, off, vl, al = [], 0.0, "v0", "a0"
        for i in range(1, n):
            off += d[i - 1] - xf
            chain.append(f"[{vl}][v{i}]xfade=transition=fade:duration={xf:.3f}:offset={off:.3f}[vx{i}];"
                         f"[{al}][a{i}]acrossfade=d={xf:.3f}[ax{i}]")
            vl, al = f"vx{i}", f"ax{i}"
        fc = norm + ";".join(chain) + f";[{vl}]null[v];[{al}]anull[a]"
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "hevc_videotoolbox", "-b:v", vbitrate, "-tag:v", "hvc1",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)


def render_sequence(clips, out, vbitrate, xfade=0.4, fps=30, sr=48000):
    """One-pass render of a highlight: clips = [(src, s, e, frames_iter, n_frames)].
    Video frames go through a single VideoToolbox encoder; the crossfade between consecutive clips is a numpy blend
    of the overlapping frames (no ffmpeg filter graph, which chained one xfade per clip and cost ~160% CPU).
    Audio of each clip is decoded to PCM, trimmed/padded to exactly its video length (no A/V drift) and crossfaded
    the same way; the two are muxed at the end."""
    import collections, os, tempfile, wave
    import numpy as np
    tmp = tempfile.mkdtemp(prefix="seq_", dir=os.environ.get("TMPDIR"))
    vid, wav = f"{tmp}/v.mov", f"{tmp}/a.wav"
    N = max(0, min(int(round(xfade * fps)), min(c[4] for c in clips) // 3)); NA = int(N * sr / fps)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080", "-r", str(fps), "-i", "-",
                            "-c:v", "hevc_videotoolbox", "-b:v", vbitrate, "-tag:v", "hvc1", "-pix_fmt", "yuv420p", vid], stdin=subprocess.PIPE)
    wf = wave.open(wav, "wb"); wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr)
    tail_v, tail_a = [], None
    for k, (src, s, e, frames, n) in enumerate(clips):
        hold = collections.deque()
        for j, f in enumerate(frames):
            if j < len(tail_v):                                          # crossfade with the previous clip's tail
                w = (j + 1) / (len(tail_v) + 1)
                f = cv2_blend(tail_v[j], f, w)
            hold.append(f)
            if len(hold) > N: enc.stdin.write(hold.popleft().tobytes())
        last = k == len(clips) - 1
        tail_v = list(hold) if not last else []
        if last:
            for f in hold: enc.stdin.write(f.tobytes())
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{s:.3f}", "-t", f"{n / fps:.3f}", "-i", src, "-vn", "-ac", "2", "-ar", str(sr),
                              "-f", "s16le", "-"], capture_output=True).stdout
        a = np.frombuffer(raw, np.int16).reshape(-1, 2).astype(np.float32)
        L = int(round(n * sr / fps)); a = np.vstack([a[:L], np.zeros((max(0, L - len(a)), 2), np.float32)])
        if tail_a is not None and NA:
            r = np.linspace(0, 1, NA, endpoint=False)[:, None]; a[:NA] = tail_a * (1 - r) + a[:NA] * r
        if not last and NA: body, tail_a = a[:-NA], a[-NA:]
        else: body = a
        wf.writeframes(np.clip(body, -32768, 32767).astype(np.int16).tobytes())
    enc.stdin.close(); enc.wait(); wf.close()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out], check=True)
    import shutil; shutil.rmtree(tmp, ignore_errors=True)


def cv2_blend(a, b, w):
    import cv2
    return cv2.addWeighted(a, 1 - w, b, w, 0)


CJK_FONTS = ["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
             "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]


def cjk_font(size):
    """A font that renders both Latin and Chinese (PingFang is not at a fixed path on recent macOS)."""
    import os
    from PIL import ImageFont
    for p in CJK_FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise FileNotFoundError("no CJK font found: " + ", ".join(CJK_FONTS))
