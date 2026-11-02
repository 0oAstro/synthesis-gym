#!/usr/bin/env bash
# Record the terminal demo to results/video/demo.{cast,gif,mp4}. Runs inside a Rex tab so it is visible live.
set -euo pipefail
cd "$(dirname "$0")/.."
task="${1:-SynthGym-v0/hard/BaFe12O19}"
out=results/video
unset AWS_PROFILE
mise exec github:asciinema/asciinema@latest -- asciinema rec --overwrite --window-size 150x44 --idle-time-limit 3 \
  -c "uv run python scripts/demo.py --task '$task' --agent claude --delay 1.6; sleep 8" "$out/demo.cast"
theme="111111,e6e6e6,1c1c1c,ff4539,b6ff3c,ffd166,4a9eff,b7a6fa,78f8a5,bcbcbc,3a3a3a,ff6b4a,c8ff6e,fcdaa4,6fb4ff,cbbefc,a0fbc0,ffffff"
mise exec github:asciinema/agg@latest -- agg --theme "$theme" --font-dir /var/tmp/synthgym-fonts/fonts/ttf --font-family "JetBrains Mono" --font-size 18 --fps-cap 20 --idle-time-limit 3 "$out/demo.cast" "$out/demo.gif"
ffmpeg -y -loglevel error -i "$out/demo.gif" -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "$out/demo.mp4"
ls -la "$out"
