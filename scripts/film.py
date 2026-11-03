"""The 60-second recut: title card, base model failing, trained model succeeding, results chart, end card.

Episodes are replayed from real evaluation transcripts (results/transcripts_<name>.json); the gym is
deterministic, so replaying the recorded tool calls reproduces the same states and XRD readings.
Usage: uv run python scripts/film.py --task SynthGym-v0/hard/X --base qwen3-4b-tools-base --trained qwen3-4b-tools-grpo
"""

import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path

import pandas as pd
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from synthgym.evaluate import wilson
from synthgym.tasks import task_spec
from synthgym.tool_env import SynthesisToolEnv

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts" / "demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
LIME, RED, BLUE, GREY = demo.LIME, demo.RED, demo.BLUE, demo.GREY


def card(live: Live, lines: list[tuple[str, str]], seconds: float) -> None:
    text = Text(justify="center")
    for content, style in lines:
        text.append(content + "\n", style=style)
    live.update(Align.center(text, vertical="middle", height=live.console.size.height), refresh=True)
    time.sleep(seconds)


def tool_calls(transcript: list[dict]) -> list[tuple[str, dict | None]]:
    """(reason, call) per assistant turn; call is None when the model produced no valid tool call."""
    turns = []
    for message in transcript:
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        reason = (message.get("content") or "").strip()
        turns.append((reason, calls[0]["function"] if calls else None))
    return turns


def replay(live: Live, task_id: str, transcript: list[dict], agent_label: str, captions: dict[int, str], delay: float) -> float:
    tool_env = SynthesisToolEnv()
    tool_env.reset(task_id=task_id)
    env = tool_env.env
    screen = demo.Screen(env, agent_label)
    screen.title = env.task.target
    screen.caption = captions.get(0, "")
    live.update(screen.render(), refresh=True)
    time.sleep(delay * 1.5)
    turn_index = 0
    for reason, call in tool_calls(transcript):
        turn_index += 1
        screen.caption = captions.get(turn_index, screen.caption)
        if call is None:
            screen.reason = Text("(no valid tool call; the model wrote prose instead)", style=RED)
            screen.action_json = "{}"
            live.update(screen.render(), refresh=True)
            time.sleep(delay)
            continue
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        screen.reason = Text(reason[:420], style="white") if reason else Text("(no commentary: the model just calls the tool)", style="dim italic")
        screen.action_json = json.dumps({"tool": call["name"], **arguments}, indent=2)
        live.update(screen.render(), refresh=True)
        time.sleep(delay * 0.6)
        getattr(tool_env, call["name"])(**arguments)
        screen.true_w[env.steps] = env.crucible.weight_fractions().get(env.task.target, 0.0)
        live.update(screen.render(), refresh=True)
        time.sleep(delay)
        if tool_env.done:
            break
    tool_env.get_reward()
    final = env.crucible.weight_fractions().get(env.task.target, 0.0)
    verdict_style = f"bold #111111 on {LIME if final >= 0.9 else RED}"
    screen.reason = Text.assemble(("  episode over  ", verdict_style), (f"  true target fraction {final:.2f}  ·  return {env.total_reward:+.2f}", "bold"))
    screen.caption = captions.get(-1, "")
    live.update(screen.render(), refresh=True)
    time.sleep(delay * 3)
    return final


def bars(rows: list[tuple[str, float, str, str]], scale: float, width: int = 46) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right")
    table.add_column()
    table.add_column(justify="right")
    for label, value, style, note in rows:
        filled = int(round(min(value / scale, 1.0) * width))
        table.add_row(Text(label, style=style), Text("█" * filled + "·" * (width - filled), style=style), Text(note, style="white"))
    return table


def chart(live: Live, agents: list[tuple[str, str, str]], seconds: float) -> None:
    success_rows = []
    temperature_rows = []
    for label, name, style in agents:
        frame = pd.read_csv(RESULTS / f"episodes_{name}.csv")
        k, n = int(frame["success_sim"].sum()), len(frame)
        low, high = wilson(k, n)
        success_rows.append((label, k / n, style, f"{k / n:.0%} [{low:.0%}, {high:.0%}]"))
        error = float(frame["temp_error_c"].dropna().mean())
        temperature_rows.append((label, error, style, f"{error:.0f} °C"))
    body = Group(
        Text("Simulated success (target ≥ 90 wt%), 200 held-out targets, Wilson 95% CI", style="bold white"),
        bars(success_rows, 1.0),
        Text(""),
        Text("Furnace temperature error vs the published recipe (lower is better)", style="bold white"),
        bars(temperature_rows, 500.0),
        Text(""),
        Text("GRPO vs base after 30 steps: success +1.5 pts (p = 0.68, not significant); temperature error 301 → 222 °C.", style=LIME),
        Text("Closed loop: agents see only paid, noisy XRD. Held-out chemical systems. Simulated outcomes.", style="dim"),
    )
    live.update(Align.center(body, vertical="middle", height=live.console.size.height), refresh=True)
    time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--base", default="qwen3-4b-tools-base")
    parser.add_argument("--trained", default="qwen3-4b-tools-grpo")
    parser.add_argument("--delay", type=float, default=1.3)
    args = parser.parse_args()
    base_transcript = json.loads((RESULTS / f"transcripts_{args.base}.json").read_text())[args.task]
    trained_transcript = json.loads((RESULTS / f"transcripts_{args.trained}.json").read_text())[args.task]
    target = task_spec(args.task).target
    console = Console()
    with Live(Text(""), console=console, screen=True, auto_refresh=False) as live:
        card(live, [("Can a model learn synthesis", f"bold {LIME}"), ("from failed experiments?", f"bold {LIME}"), ("", ""),
                    ("SynthGym: a furnace simulator with hidden state, checked against real lab outcomes", "dim")], 4)
        card(live, [(f"Target: {target}", "bold white"), ("", ""), ("Qwen3-4B, before any training in the gym", BLUE),
                    ("left: what the model thinks and does   ·   right: the true crucible (hidden) and the noisy XRD it paid for", "dim")], 3.5)
        replay(live, args.task, base_transcript, "Qwen3-4B base", {
            0: "BEFORE: base model, never practised. The published recipe uses about 900 °C.",
            3: "BEFORE: XRD shows no target, so it turns the furnace up. Hotter is not better.",
            -1: "BEFORE: 1600-1700 °C decomposes the phase. Return = purity minus furnace and XRD costs.",
        }, args.delay)
        card(live, [("Same model after GRPO in SynthGym", f"bold {LIME}"), ("", ""),
                    ("It practised on 900+ training targets and learned from every failed attempt", "white"),
                    ("(no held-out target, and no chemical system near it, was seen in training)", "dim")], 4)
        replay(live, args.task, trained_transcript, "Qwen3-4B + GRPO", {
            0: "AFTER: same model after GRPO in the gym. Same target, same tools, same noisy XRD.",
            3: "AFTER: it reads the XRD, regrinds, and reheats gently instead of escalating",
            -1: "AFTER: target reached near the published temperature, in simulation",
        }, args.delay)
        chart(live, [
            ("Qwen3-4B base", "qwen3-4b-tools-base", BLUE),
            ("Qwen3-4B + GRPO", "qwen3-4b-tools-grpo", f"bold {LIME}"),
            ("heuristic", "heuristic", GREY),
            ("Claude Opus 5.5", "claude", "white"),
        ], 9)
        card(live, [("Real labs generate these failures daily.", f"bold {LIME}"), ("We capture them.", f"bold {LIME}"), ("", ""),
                    ("github.com/0oAstro/synthesis-gym", "dim")], 4)


if __name__ == "__main__":
    main()
