#!/usr/bin/env bash
# Record the 60-second recut to results/video/film.{cast,gif,mp4}.
set -euo pipefail
cd "$(dirname "$0")/.."
out=results/video
uv run python -c "import synthgym.tasks as t; t.load_world()" >/dev/null 2>&1
mise exec github:asciinema/asciinema@latest -- asciinema rec --overwrite --window-size 150x44 --idle-time-limit 10 \
  -c "uv run python -W ignore scripts/film.py --task 'SynthGym-v0/medium/Sr2ErRuO6' --base film-base --trained qwen3-4b-tools-grpo --delay 1.1" "$out/film.cast"
theme="111111,e6e6e6,1c1c1c,ff4539,b6ff3c,ffd166,4a9eff,b7a6fa,78f8a5,bcbcbc,3a3a3a,ff6b4a,c8ff6e,fcdaa4,6fb4ff,cbbefc,a0fbc0,ffffff"
python3 scripts/trim_cast.py "$out/film.cast" "$out/film.trimmed.cast"
mise exec github:asciinema/agg@latest -- agg --theme "$theme" --font-dir /var/tmp/synthgym-fonts/fonts/ttf --font-family "JetBrains Mono" --font-size 18 --fps-cap 20 --idle-time-limit 10 "$out/film.trimmed.cast" "$out/film.gif"
ffmpeg -y -loglevel error -i "$out/film.gif" -movflags faststart -pix_fmt yuv420p -vf "fps=20,scale=trunc(iw/2)*2:trunc(ih/2)*2" -r 20 "$out/film.mp4"
ffprobe -v error -show_entries format=duration -of csv=p=0 "$out/film.mp4"
