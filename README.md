# player-highlights

Find one basketball player in your game videos — by **jersey number and colour** — and get, automatically:

- 🎬 **Highlight reel** – the player's possessions, passes and shots; original framing and/or a follow-cam zoom
  (1.2× / 1.6× / 2.0×), with clip times, a running game clock, transitions and a stats card at the end
- 📼 **Full on-court cut** – every stretch the player was on the court
- ⏱️ **Minutes played** – quarters or halves, stopped or running clock
- ✂️ **Editable clip list** – nudge, drop or add clips with one line of text

Everything runs **locally on an Apple Silicon Mac**. No footage is uploaded anywhere.

## The easy way: ask your AI assistant

Open **Claude Code** or **Codex** and paste:

> Install https://github.com/colakang/player-highlights and make highlights of my kid (number __, __ jersey) from the videos in ~/Movies/game1

The assistant installs everything, asks you a few questions (which file is which quarter, game format, zoom), runs it
in the background and tells you when your videos are ready. Problems or ideas? Just tell the assistant —
it can open an issue or a pull request for you.

## For AI assistants (Claude Code, Codex, …)

If a user asks you to install or use this repository:

1. **Check the machine**: macOS on Apple Silicon (`uname -m` = `arm64`); ~15 GB free disk for models plus ~6 GB
   temporary space per game. Intel Macs, Windows and Linux are not supported yet — tell the user.
2. **Install** (ask before installing Homebrew if it is missing — https://brew.sh):
   ```bash
   git clone https://github.com/colakang/player-highlights.git ~/player-highlights
   cd ~/player-highlights && ./setup.sh          # installs ffmpeg/uv via Homebrew if needed, Python env, models
   ```
   No API keys are needed. Optional: `./setup.sh --prefetch` downloads the vision models (~7 GB) up front.
3. **Use**: follow the skill [`.agents/skills/player-highlights/SKILL.md`](.agents/skills/player-highlights/SKILL.md)
   (also linked at `.claude/skills/`). It lists what to ask the user and how to run and report.
4. **Update** later with `git pull && ./setup.sh`.
5. **Report problems / suggest ideas**: with the user's OK, open an issue using the templates in `.github/`
   (`gh issue create --repo colakang/player-highlights`), including the output of `scripts/diag.sh`.
   Never attach videos, frames or screenshots — the players are minors; describe times instead (e.g. "Q3 09:51").
6. **Contribute code**: fork (`gh repo fork --clone`), branch, change, test with `pf.py` on a real game, then
   `gh pr create` following `.github/pull_request_template.md` (say how you measured the improvement).
   See [AGENTS.md](AGENTS.md) for the code layout and rules.

## Requirements

- Mac with an Apple M-series chip (M1 or newer), 16 GB memory recommended
- ~15 GB disk for models; ~6 GB temporary per game (cleared automatically)
- Game videos: one file per quarter / half, or one full-game file. A steady camera and sharp video help a lot
  (4K, fast shutter, camera high and centred).

## Manual use (command line)

```bash
cd ~/player-highlights
.venv/bin/python pf.py \
  --videos ~/Movies/game1 \            # a folder (sorted by file name) or the files in game order
  --periods Q1,Q2,Q3,Q4 \              # one per file: Q1..Q4 / H1,H2 / FULL
  --format quarters:8:stop \           # quarters|halves|full : minutes per period : stop|running
  --number <your kid's number> --color <jersey colour> --name <label, e.g. 7_ALEX> \
  --zooms 1,1.6                        # 1 = original framing; any of 1.2 / 1.6 / 2.0
```

Results go to `~/Movies/player-highlights/<game folder>/` (change with `--out`, a network share works too):

| File | Content |
|---|---|
| `<name>_highlight.mp4` | highlight reel, original framing |
| `<name>_highlight_zoom1.6.mp4` | highlight reel, 1.6× follow-cam |
| `*_clips.txt` | clip list |
| `<name>_full_game.mp4` | full on-court cut |

About 40–60 minutes of processing per 15 minutes of video. Re-running the same command resumes where it stopped.

### Adjusting clips

Edit `*_clips.txt` — each line is a clip (period, start time, then start/end adjustments in seconds):

```
Q3 09:41 -2 +3    # start 2 s earlier, end 3 s later
Q2 05:10 x        # drop this clip
Q4 10:20 10:35 +  # new line: add a clip
```

Then re-run the same command with `--adjust <that file>` — only the cutting is redone (about a minute).

### Your own picks: cut exactly the moments you choose

Watching the game anyway? Write down the moments (video time of each file) and get exactly those clips, captions
and a box score — in about a minute once the game has been analysed:

```
Q1: 6:39 on; 8:06 7np catch and shoot; 9:48 7p1; 12:40 4p7
Q2: 3:10; 5:03 7jb; 8:36 7m2 layup; 10:36 7reb; 10:37 7p20 to
H2: 4:15 great defence on the press
```

`7` = your kid's number. Codes: `7m2`/`7m3`/`7m1` made 2/3/free throw, `7np`/`7np1` missed shot/free throw, `7pN` passes
to #N, `Np7` #N passes to your kid, `7reb`, `7st` steal, `7t` turnover (or `to` after a pass), `7f` foul, `7gf` drew a
foul, `7screen`, `7jb` jump ball, `on`/`off`. A time with no code (or just words) is simply a clip with that caption;
`3:10~3:25` gives a time range.

```bash
.venv/bin/python pf.py ...same options... --plays my_plays.txt [--auto] [--lang en|zh]
```

Outputs `<name>_plays.mp4` (one per zoom) and `<name>_summary.md` (box score, minutes, play list). `--auto` also
makes the automatic reel; `--adjust` works here too. Your AI assistant can write this file from what you tell it
("at 8:36 in Q2 they scored a layup").

## Improve the detector with your own labels (and share the result)

The default detector is trained on a public dataset; frames from *your* gym, camera and league make it better —
especially for the ball and jersey numbers. Everything stays on your Mac; only the trained **weights** can be shared.

1. **Pick frames** from games you have already analysed (about 40 per game; keep one game for testing):
   `.venv/bin/python tools/label_frames.py --games work/<game1> work/<game2> --test work/<game3>`
2. **Correct the pre-labels** in a local Label Studio (http://localhost:8080):
   `.venv/bin/python tools/labelstudio.py start`, sign up locally, copy *Account & Settings → Personal Access Token*, then
   `.venv/bin/python tools/labelstudio.py import --token <token>`. Fix the **number** (red) and **ball** (orange) boxes,
   glance at players/referees, press *Submit* per frame (about a minute each).
3. **Export and train** (~2–4 h on the Apple GPU):
   `.venv/bin/python tools/labelstudio.py export --token <token>` then `.venv/bin/python tools/finetune.py`.
   It mixes in the public dataset, starts from the current detector and prints a before/after table on your test frames.
   If "after" is better: `.venv/bin/python tools/finetune.py --install` (the old model is kept as `weights_prev.onnx`).
4. **Share**: open an issue "Detector fine-tune" with the before/after table, how many frames / games / gyms, and attach
   `best.pt` (or a link). Never attach the labelled images. Good fine-tunes become the next `detector-vN` release, and the
   next family can start from it with `tools/finetune.py --start <file.pt>` — so every family's labels add up.

Model formats: the detector runs as **ONNX** through onnxruntime's CoreML provider (Apple GPU / Neural Engine);
training uses the **PyTorch `.pt`** (both in the `detector-v2` release). The vision-language models (Qwen2.5-VL) run
in **MLX** format, EdgeTAM in PyTorch on MPS.

## Accuracy (honest numbers)

On five youth games (1080p, sideline camera):

- On-court stretches: 93–100 % recall and precision against hand-checked times in most quarters
- Player identified for 65–85 % of their on-court time
- Highlight reel: about 3 out of 4 of the player's possessions; far-court plays, crowded rebounds and off-ball
  defence are the usual misses

Video quality matters most — 4K, a fast shutter (1/250 s or faster) and a high, centred camera all help.

## How it works

Detection (YOLO11s, CoreML) → tracking (McByte, BoT-SORT + OSNet) → jersey number reading (Qwen2.5-VL 3B / 7B,
MLX) → whistle / vision segmentation + HMM for on-court time → lineup-constrained identity → EdgeTAM mask tracking
anchored on number reads → zoomed ball detection + Qwen check of touches → shot candidates → hardware-encoded cuts.
Details in [AGENTS.md](AGENTS.md).

## 中文简介

按球衣号码和颜色，从比赛录像中自动找出指定球员，产出精华片段、完整在场片段和上场时间，全部在 M 系列芯片的 Mac
上本地运行，不上传任何画面。最简单的用法：在 Claude Code 或 Codex 里贴上本仓库链接，告诉它号码、球衣颜色和录像
位置，它会自动安装、确认细节并运行。

## Contributing

Issues and pull requests are welcome — your AI assistant can file them for you. Please never include game footage,
frames, screenshots or labels (minors). For accuracy changes, include how you measured before and after.

## Licence

[AGPL-3.0](LICENSE). Third-party components and model licences: [NOTICE.md](NOTICE.md). The default detector
(`detector-v2`) was trained on the Roboflow Universe dataset "basketball-player-detection-3" (CC BY 4.0) plus
hand-corrected youth-game frames (not published) — see the release notes for its accuracy.
