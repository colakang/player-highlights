#!/bin/bash
# One-time setup on an Apple Silicon Mac: Python env, EdgeTAM, models.
set -e
cd "$(dirname "$0")"
[ "$(uname -s)/$(uname -m)" = "Darwin/arm64" ] || { echo "player-highlights needs an Apple Silicon Mac (any M-series chip)."; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg missing: brew install ffmpeg"; exit 1; }
command -v uv >/dev/null || { echo "uv missing: brew install uv   (or: curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }
[ -d .venv ] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
if [ ! -d third_party/EdgeTAM ]; then
  git clone --depth 1 https://github.com/facebookresearch/EdgeTAM.git third_party/EdgeTAM
  # make a reshape contiguous so it runs on Apple MPS (same fix as McByte++)
  sed -i '' 's/latents_2d = self.latents_2d.unsqueeze(0).expand(B, -1, -1).view(-1, 1, C)/latents_2d = self.latents_2d.unsqueeze(0).expand(B, -1, -1).contiguous().view(-1, 1, C)/' third_party/EdgeTAM/sam2/modeling/perceiver.py
fi
.venv/bin/python scripts/setup_models.py "$@"
echo
echo "Ready. Example:"
echo "  .venv/bin/python pf.py --videos ~/Movies/game1 --periods Q1,Q2,Q3,Q4 --format quarters:8:stop --number 12 --color white --zooms 1,1.6"
