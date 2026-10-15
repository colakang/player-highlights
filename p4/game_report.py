"""Post-game report from the user's play-by-play: per-period minutes, box score, play list, stats card json,
highlight label json (plays as 'extra' clips), Markdown summary. English or Chinese (--lang).

python p4/game_report.py --plays plays.json --games Q1=work/<game>/q1 ... \
    --title "<game title>" --out-md summary.md --stats-json stats.json --labels-json hl_labels.json \
    [--qlen 8] [--clock stop|running] [--lang en|zh]
"""
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plays import stats

ap = argparse.ArgumentParser()
ap.add_argument("--plays", required=True); ap.add_argument("--games", nargs="+", required=True)
ap.add_argument("--title", required=True); ap.add_argument("--name", default="player"); ap.add_argument("--qlen", type=float, default=7.0)
ap.add_argument("--clock", choices=["stop", "running"], default="stop"); ap.add_argument("--lang", choices=["en", "zh"], default="en")
ap.add_argument("--out-md", required=True); ap.add_argument("--stats-json", required=True); ap.add_argument("--labels-json", required=True)
ap.add_argument("--pre", type=float, default=3.0); ap.add_argument("--post", type=float, default=2.5)
a = ap.parse_args()
EN = a.lang == "en"
T = (lambda en, zh: en) if EN else (lambda en, zh: zh)
P = json.load(open(a.plays)); ts = lambda t: f"{int(t) // 60:02d}:{int(t) % 60:02d}"
mmss = lambda s: f"{int(s) // 60}:{int(round(s)) % 60:02d}"

rows, labels, allp, tot_min = [], {}, [], 0.0
for spec in a.games:
    q, W = spec.split("=")
    tl = json.load(open(f"{W}/p3/timeline.json")); dur = json.load(open(f"{W}/meta.json"))["duration"]
    on = sum(e - s for s, e in tl["intervals"]); dead = sum(e - s for s, e in tl.get("dead", []))
    mins = on / 60 if a.clock == "running" else a.qlen * on / max(1.0, dur - dead); tot_min += mins   # game-clock minutes
    ps = sorted([p for p in P.get(q, []) if not p["onoff"]], key=lambda p: p["t"]); allp += ps
    rows.append((q, tl["intervals"], mins, stats(ps), ps))
    labels[q] = dict(work=W, candidates=[], extra=[dict(t=max(0, p["t"] - a.pre), t_end=p["t_end"] + a.post, her_zh=p["caption"])
                                                    for p in ps if p["caption"] is not None])
S = stats(allp)
json.dump(labels, open(a.labels_json, "w"), indent=1, ensure_ascii=False)
fg = f"{S['FGM']}/{S['FGA']}"; ft = f"{S['FTM']}/{S['FTA']}"
card_rows = [[T("Points", "得分"), str(S["PTS"])], [T("Field goals", "投篮"), fg]] + ([[T("Free throws", "罚球"), ft]] if S["FTA"] else []) + \
            [[T("Rebounds", "篮板"), str(S["REB"])], [T("Steals", "抢断"), str(S["STL"])], [T("Passes", "传球"), str(S["PASS"])],
             [T("Turnovers", "失误"), str(S["TOV"])], [T("Screens", "掩护"), str(S["SCREEN"])]]
json.dump(dict(title=a.name, subtitle=a.title, minutes=mmss(tot_min * 60), minutes_label=T("Minutes", "上场时间"), rows=card_rows,
               note=T("Minutes estimated from the video (±1 min); stats from hand-written play-by-play",
                      "上场时间按视频估算（±1 分钟）；数据来自人工逐回合标注")),
          open(a.stats_json, "w"), indent=1, ensure_ascii=False)

clock_txt = T(f"{int(a.qlen)}-min periods, {'running' if a.clock == 'running' else 'stopped'} clock",
              f"每节 {int(a.qlen)} 分钟{'，running clock' if a.clock == 'running' else '停表'}")
L = [f"# {a.name} · {a.title}", "",
     f"**{T('Minutes played', '上场时间')} ≈ {mmss(tot_min * 60)}** ({clock_txt}; ±1 min)", "",
     f"## {T('Box score', '全场数据')}", "",
     T("| PTS | FG | FT | REB | STL | PASS | CATCH | TOV | SCREEN | JUMP BALL |", "| 得分 | 投篮 | 罚球 | 篮板 | 抢断 | 传球 | 接球 | 失误 | 掩护 | 争球 |"),
     "|---|---|---|---|---|---|---|---|---|---|",
     f"| {S['PTS']} | {fg} | {ft} | {S['REB']} | {S['STL']} | {S['PASS']} | {S['RECV']} | {S['TOV']} | {S['SCREEN']} | {S['JB']} |", "",
     f"## {T('By period', '分节')}", "",
     T("| Period | On court (video time) | Min | PTS | FG | REB | STL | PASS | TOV |", "| 节 | 在场时段（视频时间） | 上场 | 得分 | 投篮 | 篮板 | 抢断 | 传球 | 失误 |"),
     "|---|---|---|---|---|---|---|---|---|"]
for q, iv, mins, s, ps in rows:
    L.append(f"| {q} | {', '.join(f'{ts(x)}–{ts(y)}' for x, y in iv) or T('did not play', '未上场')} | {mmss(mins * 60)} | {s['PTS']} | "
             f"{s['FGM']}/{s['FGA']} | {s['REB']} | {s['STL']} | {s['PASS']} | {s['TOV']} |")
L += ["", f"## {T('Plays', '逐回合')}", ""]
for q, iv, mins, s, ps in rows:
    if not ps: continue
    L.append(f"**{q}**"); L.append("")
    for p in ps: L.append(f"- {ts(p['t'])}{('–' + ts(p['t_end'])) if p['t_end'] > p['t'] else ''}  {p['caption'] or p['raw']}")
    L.append("")
obs = []
if S["TOV"] >= 3:
    tg = collections.Counter(p.get('target') for p in allp if p['turnover'] and p.get('target')).most_common(1)
    many = tg and tg[0][1] >= 2
    obs.append(T(f"{S['TOV']} turnovers" + (f", mostly on passes to #{tg[0][0]}" if many else "") + " - worth reviewing pass choice and timing.",
                 f"失误 {S['TOV']} 次" + (f"，多为传给 {tg[0][0]} 号的球被断" if many else "") + "，传球选择和时机值得复盘。"))
if S["REB"]: obs.append(T(f"Active on the glass: {S['REB']} rebounds" + (f", {S['STL']} steals" if S["STL"] else "") + ".",
                          f"拼抢积极：{S['REB']} 个篮板" + (f"、{S['STL']} 次抢断" if S["STL"] else "") + "。"))
if S["FGA"]: obs.append(T(f"{S['FGM']} of {S['FGA']} from the field.", f"出手 {S['FGA']} 次命中 {S['FGM']} 次。"))
if S["SCREEN"]: obs.append(T(f"{S['SCREEN']} screens set - good team play.", f"无球掩护 {S['SCREEN']} 次，团队配合意识不错。"))
if obs: L += [f"## {T('Notes', '要点')}", ""] + [f"- {o}" for o in obs]
os.makedirs(os.path.dirname(os.path.abspath(a.out_md)), exist_ok=True)
open(a.out_md, "w").write("\n".join(L) + "\n")
print("\n".join(L))
