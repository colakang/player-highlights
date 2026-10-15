#!/bin/bash
# One-time setup on an Apple Silicon Mac: Python env, EdgeTAM, models.
set -e
cd "$(dirname "$0")"
[ "$(uname -s)/$(uname -m)" = "Darwin/arm64" ] || { echo "player-highlights needs an Apple Silicon Mac (any M-series chip)."; exit 1; }
# command-line tools: install what is missing with Homebrew (https://brew.sh)
for tool in ffmpeg uv git; do
  if ! command -v $tool >/dev/null; then
    if command -v brew >/dev/null; then echo "installing $tool (Homebrew) ..."; brew install $tool
    else echo "$tool is missing and Homebrew is not installed. Install Homebrew first: https://brew.sh"; exit 1; fi
  fi
done
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
