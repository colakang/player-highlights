---
name: player-highlights
description: Make a basketball player's highlight reel, full on-court cut and minutes played from game videos, by jersey number and colour, fully locally on an Apple Silicon Mac. Use when the user gives game video files or a folder and wants their child's / a player's highlights, minutes, or on-court clips.
---

# player-highlights

Turns game videos into: an automatic highlight reel (one per chosen zoom level), the full on-court version,
minutes played, and a clip list the user can edit. Everything runs locally (no uploads). Repo root = the folder
containing `pf.py`.

## 1. Install / check setup (once per machine)

- Needs macOS on Apple Silicon (`uname -m` = `arm64`, any M-series chip), ~15 GB free disk for models plus ~6 GB
  temporary space per game. Intel Macs, Windows and Linux are not supported yet - say so and stop.
- If the repo is not on the machine: `git clone https://github.com/colakang/player-highlights.git ~/player-highlights`.
- If `.venv/`, `models/detector/weights.onnx` or `third_party/EdgeTAM/` is missing, run `./setup.sh` from the repo root.
  It installs ffmpeg / uv / git with Homebrew when missing (if Homebrew itself is missing, ask the user before
  installing it from https://brew.sh), the Python environment, EdgeTAM and the detector. No API keys are needed.
  Optional `./setup.sh --prefetch` downloads the Qwen2.5-VL models (~7 GB) up front; otherwise they download on first use.
- To update: `git pull && ./setup.sh`.

## 2. Collect the inputs - ask the user, never guess

Ask in the user's language, in one message where possible:

1. **Videos**: a folder or files. Videos are read in place (local disk, external drive or network share all work).
   List what you find (name + duration via `ffprobe -v error -show_entries format=duration -of csv=p=0 FILE`).
2. **Which period each file is** - confirm explicitly: Q1..Q4, H1/H2, or FULL. Do not infer from durations alone;
   a long gap inside a quarter is not halftime.
3. **Players per side**: 5 (default), or 3 / 4 for small-sided games (half-court 4v4 -> `--team-size 4`).
   **Game format and clock**: e.g. 4 quarters x 8 min stopped clock (`quarters:8:stop`), 2 halves x 20 min
   running clock (`halves:20:running`). Used for minutes played.
4. **Jersey number and team jersey colour** of the player.
5. **Name label** for captions and file names (e.g. `7_ALEX`; default is the number).
6. **Zoom levels** - multiple choice from 1 (original framing), 1.2, 1.6, 2.0; default 1. Each level is one reel.
7. **Output folder** - default `~/Movies/player-highlights/<game folder>/`. Must be writable.
8. **Language** of captions / stats card / report: English (default) or Chinese (`--lang zh`).
9. **Own picks?** If the user has (or wants to dictate) the moments to include, write them into a plays file
   (format in README "Your own picks") and pass `--plays <file>` (add `--auto` to also get the automatic reel).
10. Optional: a notification command (e.g. a chat CLI) - pass as `--notify '... "{msg}"'`.

## 3. Run

It takes about 40-60 minutes per 15 minutes of video. Start it detached so it survives the session, then report progress
from the log:

```bash
nohup .venv/bin/python pf.py --videos "<folder or files>" --periods Q1,Q2,Q3,Q4 --format quarters:8:stop \
  --number <N> --color <colour> --name <label> --zooms 1,1.6 --out "<output folder>" > work/pf_<game>.log 2>&1 &
```

- Re-running the same command continues where it stopped (finished periods are skipped).
- Watch the log for `=== Q2`, `done (…s)` and the final summary; on failure the log names the step.

## 4. Report back

From the final summary in the log: output files, minutes played, and the on-court stretches per period
(video time). Ask the user to check the on-court stretches; very short ones (2-3 s) are usually false.

## 5. Adjust clips (optional)

Next to each reel is `<reel>_clips.txt`, one line per clip (`Q3 09:41 +0 +0    # -> 10:48  67s`). The user edits it
in place (best on a copy):
- `+0 +0` -> `-2 +3` - start 2 s earlier, end 3 s later
- `+0 +0` -> `x` - drop the clip
- a new line `Q3 10:20 10:35 +` - add a clip.

Then re-run the same `pf.py` command with `--adjust <edited file>`; only the cutting is redone.

## Notes

- Accuracy limits (honest): the player is identified for ~65-85% of their on-court time; the automatic reel catches
  roughly 3 out of 4 of their ball possessions. Far-court plays, crowded rebounds and off-ball defence are the
  usual misses. Sharper video (4K, fast shutter, higher camera) helps most.
- Videos of minors: keep all outputs and the `work/` folder private; nothing here uploads anything.
- Basketball only for now.

## 6. Better detector from the user's own labels (optional)

If the user wants to improve accuracy for their gym/camera, follow README "Improve the detector": `tools/label_frames.py`,
`tools/labelstudio.py start|import|export` (the user corrects boxes in http://localhost:8080 and gives you the
Personal Access Token), then `tools/finetune.py` (2-4 h on the GPU; suggest running it when the Mac is free) and
`--install` only if the before/after table improves. Offer to share the weights + table via an issue - never images.

## 7. Problems, ideas, contributions

- Something failed or a result looks wrong: run `scripts/diag.sh`, show the user what will be sent, and with their OK
  open an issue: `gh issue create --repo colakang/player-highlights --title "..." --body "..."` following
  `.github/ISSUE_TEMPLATE/problem.md`. If `gh` is missing: `brew install gh && gh auth login` (the user logs in).
- Never attach videos, frames, screenshots or label files - the players are minors. Refer to moments by time
  ("Q3 09:51") instead.
- Ideas: `.github/ISSUE_TEMPLATE/idea.md`.
- Code changes: `gh repo fork colakang/player-highlights --clone`, work on a branch, test with `pf.py` on a real game,
  then `gh pr create` using `.github/pull_request_template.md` with before/after numbers. Layout and rules: `AGENTS.md`.
