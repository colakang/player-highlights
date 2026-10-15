"""Time-based play-by-play (what the user writes while watching) -> plays json, stats, captions.

Input text, one line per period (Q1..Q4 / H1 H2 / FULL; times = video time of that period's file),
entries separated by ';' (or just by the next time). A time with no code (e.g. "Q2: 3:10; 7:45 nice defence") is simply
a clip to include, captioned with any text after it:
  Q2: 0:43 30p12; 0:48 12p 0:44~0:48 12运球; 0:52 12Reb; 8:36 12np(catch and shot); 10:08 12p15
  Q3: 0:41 12p43,to; 9:51 12m2,layup; 12:30 off; 16:40 on; 16:48 12 跳起防守
  Q4: 11:57 on; 12:31 12st
Vocabulary (case-insensitive; '12' = target number):
  12np / 12np1 / 12m1 12m2 12m3     shot missed / FT missed / made 1-2-3
  12p  12pN  Np12                   #12 passes (to N) / N passes to #12
  ... to | ,to                      the pass/play was a turnover
  12st 12t 12r|12reb 12f 12gf 12screen 12jumball|12jb   steal, turnover, rebound, foul, drew foul, screen, jump ball
  on / off                          #12 enters / leaves (on-court truth)
  anything else (e.g. 运球, 跳起防守, layup, catch and shot) -> kept as caption text
"""
import json, re, sys

NUM = "12"
LANG = "en"
EN = {"np": "missed shot", "np1": "missed free throw", "m1": "free throw made", "m2": "2 points", "m3": "3 points", "st": "steal",
      "t": "turnover", "r": "rebound", "reb": "rebound", "f": "foul", "gf": "drew a foul", "screen": "screen", "jumball": "jump ball",
      "jb": "jump ball"}
NOTE_EN = {"layup": "layup", "catch and shoot": "catch and shoot", "catch and shot": "catch and shoot", "to": "turnover"}
ZH = {"np": "出手未进", "np1": "罚球未进", "m1": "罚球命中", "m2": "两分命中", "m3": "三分命中", "st": "抢断", "t": "失误",
      "r": "篮板", "reb": "篮板", "f": "犯规", "gf": "造犯规", "screen": "掩护", "jumball": "争球", "jb": "争球"}
NOTE_ZH = {"layup": "上篮", "catch and shot": "接球投篮", "to": "失误"}


def _t(s):
    m, x = s.split(":"); return int(m) * 60 + float(x)


def parse(text, number=NUM, lang="en"):
    global NUM, LANG; NUM = number; LANG = lang
    out = {}
    for line in text.strip().splitlines():
        m = re.match(r"\s*(Q\d|H\d|FULL)\s*[:：]\s*(.*)", line.strip(), re.I)
        if not m: continue
        q, body = m.groups(); q = q.upper(); plays = out.setdefault(q, [])
        # split at every time token (mm:ss or mm:ss~mm:ss)
        parts = re.split(r"(\d{1,2}:\d{2}(?:\s*[~\-－]\s*\d{1,2}:\d{2})?)", body)
        for i in range(1, len(parts), 2):
            tok = parts[i]; rest = parts[i + 1].strip(" ;；,，")
            if "~" in tok or re.search(r"\d\s*[-－]\s*\d", tok):
                a, b = re.split(r"\s*[~\-－]\s*", tok); t0, t1 = _t(a), _t(b)
            else:
                t0 = t1 = _t(tok)
            plays.append(dict(t=t0, t_end=t1, raw=rest, **interpret(rest)))
    return out


def interpret(raw):
    s = raw.strip(); low = s.lower()
    d = dict(by12=False, to12=False, act=None, target=None, passer=None, turnover=bool(re.search(r"(^|[\s,，])to\b", low)),
             onoff=None, notes=[])
    if low in ("on", "off"): d.update(onoff=low, caption=None); return d
    m = re.search(rf"(\d*)p{NUM}\b", low)
    if m and not low.startswith(f"{NUM}p"): d.update(to12=True, act="p_in", passer=m.group(1) or None)
    m = re.search(rf"\b{NUM}p(\d*)\b", low)
    if m: d.update(by12=True, act="p_out", target=m.group(1) or None)
    m = re.search(rf"\b{NUM}\s*(np1|np|m1|m2|m3|st|reb|r|gf|f|t|screen|jumball|jb)\b", low)
    if m: d.update(by12=True, act=m.group(1))
    if d["act"] is None and low.startswith(NUM):                 # free text by the player, e.g. "12 blocks the shot"
        d.update(by12=True, act="text")
    elif d["act"] is None and not d["to12"]:                     # a plain time (+ optional text): just a clip to include
        d.update(by12=True, act="clip")
    for k, v in (NOTE_EN if LANG == "en" else NOTE_ZH).items():
        if k in low and k != "to": d["notes"].append(v)
    zh_free = re.sub(rf"^{NUM}\s*", "", s) if d["act"] in ("text", "clip") else ""
    d["caption"] = caption(d, zh_free)
    return d


def caption(d, zh_free=""):
    if d["onoff"]: return None
    en = LANG == "en"
    if d["act"] == "p_in": c = (f"pass from #{d['passer']}" if en else f"接 {d['passer']} 号传球") if d["passer"] else ("catch" if en else "接球")
    elif d["act"] == "p_out": c = (f"pass to #{d['target']}" if en else f"传给 {d['target']} 号") if d["target"] else ("pass" if en else "传球")
    elif d["act"] in ("text", "clip"): c = zh_free
    else: c = (EN if en else ZH).get(d["act"], d["act"] or "")
    if d["turnover"]: c += " · turnover" if en else " · 失误"
    if d["notes"]: c += " · " + " · ".join(d["notes"])
    return c or ("" if d["act"] == "clip" else None)


def stats(plays):
    s = dict(PTS=0, FGM=0, FGA=0, FTM=0, FTA=0, REB=0, STL=0, TOV=0, PASS=0, RECV=0, SCREEN=0, JB=0, DRAWN=0, PF=0)
    for p in plays:
        a = p["act"]
        if a in ("m2", "m3"): s["FGM"] += 1; s["FGA"] += 1; s["PTS"] += int(a[1])
        elif a == "np": s["FGA"] += 1
        elif a == "m1": s["FTM"] += 1; s["FTA"] += 1; s["PTS"] += 1
        elif a == "np1": s["FTA"] += 1
        elif a in ("r", "reb"): s["REB"] += 1
        elif a == "st": s["STL"] += 1
        elif a == "screen": s["SCREEN"] += 1
        elif a in ("jumball", "jb"): s["JB"] += 1
        elif a == "gf": s["DRAWN"] += 1
        elif a == "f": s["PF"] += 1
        elif a == "t": s["TOV"] += 1
        if a == "p_out": s["PASS"] += 1
        if a == "p_in": s["RECV"] += 1
        if p["turnover"] and p["by12"]: s["TOV"] += 1
    return s


if __name__ == "__main__":
    txt = open(sys.argv[1]).read(); P = parse(txt, *(sys.argv[3:5]))
    json.dump(P, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    for q, ps in P.items():
        print(q, stats(ps))
        for p in ps: print("   ", f"{int(p['t'])//60}:{int(p['t'])%60:02d}", p["raw"], "->", p["caption"], "| on/off" if p["onoff"] else "")
