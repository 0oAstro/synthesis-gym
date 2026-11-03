"""Trim an asciinema v3 cast to the full-screen app: drop the dead start before the alternate screen
opens and everything after it closes. Usage: python scripts/trim_cast.py in.cast out.cast"""

import json
import sys

ENTER = "\x1b[?1049h"
LEAVE = "\x1b[?1049l"

lines = open(sys.argv[1]).read().splitlines()
header, events = lines[0], [json.loads(line) for line in lines[1:]]
started = False
kept = []
for interval, kind, data in events:
    if not started:
        if ENTER in data:
            started = True
            kept.append([0.0, kind, data[data.index(ENTER):]])
        continue
    if LEAVE in data:
        kept.append([interval, kind, data[: data.index(LEAVE)]])
        kept.append([2.0, kind, ""])
        break
    kept.append([interval, kind, data])
with open(sys.argv[2], "w") as handle:
    handle.write(header + "\n")
    for event in kept:
        handle.write(json.dumps(event) + "\n")
print(f"kept {len(kept)} of {len(events)} events, {sum(event[0] for event in kept):.1f} s")
