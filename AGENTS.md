# AGENTS.md

Guide for coding agents (Codex, Claude Code, …) working in this repository.

## What this is

`player-highlights` finds one basketball player (jersey number + team colour) in game videos and produces a
highlight reel, a full on-court cut and minutes played - locally on Apple Silicon. The user-facing workflow is the
skill in `.agents/skills/player-highlights/SKILL.md` (also linked at `.claude/skills/`). Follow it when a user asks
for highlights.

## Layout

- `pf.py` - one-command entry point (per period: `run.py` + `p4/*`; then reels, full cut, stats card)
- `run.py` - per-period analysis: frames (6 fps), detection, tracking, number reading, on-court timeline
- `p1/` detection (YOLO ONNX on CoreML) and tracking (McByte, BoT-SORT/OSNet), number reads (Qwen2.5-VL-3B)
- `p2/` appearance features (DINOv2, colour stripes)
- `p3/` on-court decision: whistles / vision segmentation, HMM, 1 s edge refinement, Qwen2.5-VL-7B rechecks
- `p4/` identity (lineup matching, number-anchored EdgeTAM masks), touches (+ Qwen check), shots, cutting/rendering
- `scripts/setup_models.py`, `setup.sh` - environment and model download
- `third_party/` (not committed) - EdgeTAM clone made by `setup.sh`

Scripts share a work dir through the `PF_WORK` environment variable (`pfcfg.py`). Run Python with `.venv/bin/python`
from the repo root, with `PYTORCH_ENABLE_MPS_FALLBACK=1` for EdgeTAM.

## Rules

- Never commit videos, frames, crops, labels or any data showing players (minors) - `work/` and media are ignored.
- No personal paths, names, accounts or keys in code; personal settings go in `.env` / `config.local.sh` (ignored).
- Keep everything local: no uploads of footage to cloud services.
- Hardware decode/encode (VideoToolbox) for every ffmpeg call; heavy models on MPS/MLX/CoreML.
- Measure before claiming improvements (on-court recall/precision, play coverage) and say how it was measured.
- Licence: AGPL-3.0 (BoxMOT dependency). Note third-party licences in `NOTICE.md` when adding models.
