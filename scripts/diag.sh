#!/bin/bash
# Environment + recent log summary for bug reports. Prints text only; home folder paths are replaced by ~ .
cd "$(dirname "$0")/.."
{
echo "## Environment"
echo "- macOS $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB RAM"
echo "- free disk: $(df -h . | tail -1 | awk '{print $4}')"
echo "- repo: $(git rev-parse --short HEAD 2>/dev/null) $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') local changes"
echo "- python: $([ -x .venv/bin/python ] && .venv/bin/python --version 2>&1 || echo "no .venv - run ./setup.sh")"
echo "- ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"
echo "- detector: $(ls models/detector/weights.onnx >/dev/null 2>&1 && echo ok || echo missing), EdgeTAM: $(ls third_party/EdgeTAM/checkpoints/edgetam.pt >/dev/null 2>&1 && echo ok || echo missing)"
L=$(ls -t work/*/*/pf.log work/pf_*.log 2>/dev/null | head -1)
if [ -n "$L" ]; then echo; echo "## Last log ($(basename "$(dirname "$L")")/$(basename "$L"), last 40 lines)"; echo '```'; tail -40 "$L"; echo '```'; fi
} | sed "s#$HOME#~#g"
