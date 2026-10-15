"""Follow-cam: crop/zoom a clip so the player stays centred.

render_follow(src, s, e, boxes, out_mov, caption_png, zoom=1.6)
  boxes: {time_s: (x1, y1, x2, y2)} in source pixels (6 fps samples are fine); missing -> no zoom for that clip.
Decode (VideoToolbox) -> numpy crop/resize -> HEVC VideoToolbox; audio copied from the same source range (PCM),
so the part has the same A/V layout as the other cutters' parts.
"""
import subprocess
import cv2, numpy as np

W, H = 1920, 1080


def path(boxes, s, e, fps=30, zoom=1.6, min_cover=0.5, smooth_s=1.0, max_speed=0.35):
    """per-output-frame crop rectangle (x, y, w, h) or None if the player is not tracked well enough in [s, e]."""
    ts = sorted(t for t in boxes if s <= t <= e)
    n = int(round((e - s) * fps))
    if n <= 0 or len(ts) < 2 or (ts[-1] - ts[0]) < min_cover * (e - s):
        return None
    tt = np.array(ts); B = np.array([boxes[t] for t in ts], float)
    cx = (B[:, 0] + B[:, 2]) / 2; cy = (B[:, 1] + B[:, 3]) / 2; bh = B[:, 3] - B[:, 1]
    q = s + np.arange(n) / fps
    cx = np.interp(q, tt, cx); cy = np.interp(q, tt, cy); bh = np.interp(q, tt, bh)
    k = max(1, int(smooth_s * fps)); ker = np.ones(k) / k
    pad = lambda v: np.convolve(np.r_[np.full(k, v[0]), v, np.full(k, v[-1])], ker, "same")[k:-k]
    cx, cy, bh = pad(cx), pad(cy), pad(bh)
    ch = np.clip(np.maximum(H / zoom, 3.0 * bh), H / zoom, H).astype(float)    # keep >= 3 player heights in view
    ch = pad(ch); cw = ch * W / H
    # limit pan speed (fraction of crop width per second) so the camera does not whip
    vmax = max_speed * cw / fps
    for i in range(1, n):
        cx[i] = cx[i - 1] + np.clip(cx[i] - cx[i - 1], -vmax[i], vmax[i])
        cy[i] = cy[i - 1] + np.clip(cy[i] - cy[i - 1], -vmax[i], vmax[i])
    cyc = cy - 0.08 * ch                                                        # a bit of headroom: player slightly low
    x = np.clip(cx - cw / 2, 0, W - cw); y = np.clip(cyc - ch / 2, 0, H - ch)
    return np.stack([x, y, cw, ch], 1)


def full_rects(s, e, fps=30):
    """no zoom: the whole frame for every output frame (same renderer, so the running clock is drawn too)"""
    return np.tile([0.0, 0.0, W, H], (int(round((e - s) * fps)), 1))


def follow_frames(src, s, e, rects, caption_png, fps=30, clock=None):
    """yield the output frames of one clip: VideoToolbox decode -> crop/zoom -> caption (blended only inside its
    bounding box) -> running clock "Q3 09:51" bottom-right (clock = period label, None = off)"""
    n = len(rects)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-ss", f"{s:.3f}", "-i", src, "-frames:v", str(n),
                            "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
    cap = cv2.imread(caption_png, cv2.IMREAD_UNCHANGED) if caption_png else None
    if cap is not None:
        ys, xs = np.nonzero(cap[..., 3]); cy0, cy1, cx0, cx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        alpha = cap[cy0:cy1, cx0:cx1, 3:4] / 255.0; capc = cap[cy0:cy1, cx0:cx1, :3] * alpha
    fb = W * H * 3
    for i in range(n):
        buf = dec.stdout.read(fb)
        if len(buf) < fb: break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        x, y, cw, ch = rects[i]
        if int(cw) == W and int(ch) == H: f = f.copy()
        else: f = cv2.resize(f[int(y):int(y + ch), int(x):int(x + cw)], (W, H), interpolation=cv2.INTER_LINEAR)
        if cap is not None:
            r = f[cy0:cy1, cx0:cx1]; f[cy0:cy1, cx0:cx1] = (r * (1 - alpha) + capc).astype(np.uint8)
        if clock is not None:
            t = s + i / fps; txt = f"{clock} {int(t) // 60:02d}:{int(t) % 60:02d}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
            x0, y0 = W - tw - 60, H - 50
            cv2.rectangle(f, (x0 - 20, y0 - th - 20), (x0 + tw + 20, y0 + 20), (0, 0, 0), -1)
            cv2.putText(f, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        yield f
    dec.stdout.close(); dec.kill(); dec.wait()


def render_follow(src, s, e, rects, out, caption_png, bitrate="10M", fps=30, clock=None):
    """one clip to a .mov part (video + PCM audio from the same source range)"""
    n = len(rects)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                            "-ss", f"{s:.3f}", "-t", f"{n / fps:.3f}", "-i", src, "-map", "0:v", "-map", "1:a?",
                            "-c:v", "hevc_videotoolbox", "-b:v", bitrate, "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
                            "-c:a", "pcm_s16le", out], stdin=subprocess.PIPE)
    for f in follow_frames(src, s, e, rects, caption_png, fps, clock): enc.stdin.write(f.tobytes())
    enc.stdin.close(); enc.wait()
