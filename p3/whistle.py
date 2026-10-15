"""Whistle detector: narrow-band tonal energy in 2.5-4.5 kHz that stands well above the broadband floor."""
import sys, os, json, numpy as np, scipy.io.wavfile as wf, scipy.signal as ss
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pfcfg import W
sr, x = wf.read(f"{W}/audio/game.wav"); x = x.astype(np.float32) / 32768
f, t, S = ss.stft(x, sr, nperseg=2048, noverlap=2048 - 512)        # ~23 ms hop
P = np.abs(S) ** 2 + 1e-12
band = (f >= 2500) & (f <= 4500)
peak = P[band].max(0)                                               # strongest bin in whistle band
floor = np.median(P[(f > 500) & (f < 8000)], 0)                     # broadband floor
tonal = 10 * np.log10(peak / floor)                                 # dB above floor
loud = 10 * np.log10(peak)
fpk = f[band][P[band].argmax(0)]
TON, LOUD, MIN_DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 25, float(sys.argv[2]) if len(sys.argv) > 2 else -45, 0.15
on = (tonal > TON) & (loud > LOUD)
ev, i, hop = [], 0, t[1] - t[0]
while i < len(on):
    if on[i]:
        j = i
        while j < len(on) and (on[j] or (j + 3 < len(on) and on[j + 1:j + 4].any())): j += 1
        if (j - i) * hop >= MIN_DUR:
            ev.append(dict(t=float(t[i]), dur=float((j - i) * hop), f=float(np.median(fpk[i:j])), ton=float(tonal[i:j].max()), loud=float(loud[i:j].max())))
        i = j
    i += 1
json.dump(ev, open(f"{W}/audio/whistles.json", "w"), indent=0)
print(f"{len(ev)} whistle events (tonal>{TON}dB, loud>{LOUD}dB, >= {MIN_DUR}s)")
for e in ev: print(f"  {int(e['t'])//60:02d}:{e['t']%60:04.1f}  {e['dur']:.2f}s  {e['f']:.0f}Hz  +{e['ton']:.0f}dB  {e['loud']:.0f}dB")
