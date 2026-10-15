"""Review shorthand -> structured labels, captions and box-score stats.

Codes (a leading "12" = done by the target player; generalises to any number via --number):
  np        no score (missed shot)          M1 / M2 / M3   made 1 / 2 / 3 points (M1 = made free throw)
  p         pass                            p12            pass TO #12          12p   #12 passes
  st        steal                           t              turnover
  r         rebound                         12r            #12 rebound
  gf        drew a foul                     12gf           #12 drew a foul
  f         foul                            12f            #12 fouled
Extensions (proposed): np1 / np2 / np3 = missed FT / 2 / 3;  x = neither shot nor pass;  jb = jump ball;  12st / 12t / 12np ...

Reply format, one line per quarter:
  Q1: 1 p, 3 12st, 4 12M2, 9 12np 12gf, 11 12np1, 12 12np1, 18 12jb, 19 12p, 20 x
  Q4: +01:39-01:50 12gf          <- a play the candidates missed (time range in the video)
"""
import json, re

ZH = {"np": "出手未进", "np1": "罚球未进", "np2": "两分未进", "np3": "三分未进", "M1": "罚球命中", "M2": "两分命中", "M3": "三分命中",
      "p": "传球", "st": "抢断", "t": "失误", "r": "篮板", "gf": "造犯规", "f": "犯规", "jb": "争球", "x": None, "p12": "接应传球"}
SHOT = {"np", "np2", "np3", "M2", "M3"}; FT = {"np1", "M1"}


def split(code, number="12"):
    """'12M2' -> (True, 'M2'); 'p12' -> (False, 'p12'); 'M2' -> (False, 'M2')"""
    if code == f"p{number}": return False, "p12"
    if code.startswith(number) and len(code) > len(number): return True, code[len(number):]
    return False, code


def parse(text, number="12"):
    """-> {"Q1": {"items": {k: [codes]}, "extra": [{"t": s, "t_end": e, "codes": [...]}]}}"""
    out = {}
    for line in text.strip().splitlines():
        m = re.match(r"\s*(Q\d)\s*[:：]\s*(.*)", line)
        if not m: continue
        q, body = m.groups(); d = out.setdefault(q, {"items": {}, "extra": []})
        for part in re.split(r"[,，;；]\s*", body):
            part = part.strip()
            if not part: continue
            mx = re.match(r"\+\s*(\d+):(\d+)\s*-\s*(\d+):(\d+)\s+(.*)", part)
            if mx:
                a, b, c, e, codes = mx.groups()
                d["extra"].append(dict(t=int(a) * 60 + int(b), t_end=int(c) * 60 + int(e), codes=codes.split())); continue
            mk = re.match(r"#?(\d+)\s+(.*)", part)
            if mk: d["items"][int(mk.group(1))] = mk.group(2).split()
    return out


def label_of(codes, number="12"):
    """shot-classifier label from codes"""
    base = {split(c, number)[1] for c in codes}
    if base & FT: return "ft"
    if base & SHOT or base & {"M2", "M3"}: return "shot"
    if base & {"p", "p12"}: return "pass"
    if base == {"x"}: return "none"
    return "other"


def caption(codes, number="12"):
    """Chinese caption of #target's part, or None if the target is not involved"""
    zh = []
    for c in codes:
        mine, b = split(c, number)
        if (mine or b == "p12") and ZH.get(b): zh.append(ZH[b] if b != "p" else "传球")
    return " · ".join(dict.fromkeys(zh)) or None


def stats(all_codes, number="12"):
    s = dict(FGM=0, FGA=0, M3=0, A3=0, FTM=0, FTA=0, PTS=0, REB=0, AST=0, STL=0, TOV=0, PF=0, DRAWN=0)
    for codes in all_codes:
        bases = [split(c, number) for c in codes]
        for mine, b in bases:
            if not mine: continue
            if b in ("M2", "M3"): s["FGM"] += 1; s["FGA"] += 1; s["PTS"] += int(b[1])
            if b in ("np", "np2", "np3"): s["FGA"] += 1
            if b in ("M3", "np3"): s["A3"] += 1; s["M3"] += b == "M3"
            if b == "M1": s["FTM"] += 1; s["FTA"] += 1; s["PTS"] += 1
            if b == "np1": s["FTA"] += 1
            if b == "r": s["REB"] += 1
            if b == "st": s["STL"] += 1
            if b == "t": s["TOV"] += 1
            if b == "f": s["PF"] += 1
            if b == "gf": s["DRAWN"] += 1
        # assist: #12 passes and the same play ends in a made basket by someone else
        if any(m and b == "p" for m, b in bases) and any((not m) and b in ("M2", "M3") for m, b in bases): s["AST"] += 1
    return s


if __name__ == "__main__":
    demo = "Q1: 1 p, 3 12st, 4 12M2, 9 12np 12gf, 11 12np1, 18 12jb, 19 12p, 20 x\nQ4: +01:39-01:50 12gf, 14 p12, 31 12M2"
    P = parse(demo); print(json.dumps(P, ensure_ascii=False))
    for q, d in P.items():
        for k, cs in d["items"].items(): print(q, k, cs, label_of(cs), caption(cs))
    print(stats([cs for d in P.values() for cs in d["items"].values()] + [x["codes"] for d in P.values() for x in d["extra"]]))
