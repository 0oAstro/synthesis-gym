"""Live terminal demo: one agent synthesising one target, the hidden crucible beside what the agent sees.

Usage: uv run python scripts/demo.py --task "SynthGym-v0/hard/LiBiP2O7" --agent claude --delay 1.5
"""

import argparse
import json
import time
import warnings
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from synthgym.agents import ClaudeAgent, HeuristicAgent, RandomAgent
from synthgym.env import SynthesisGym
from synthgym.tasks import task_spec

warnings.filterwarnings("ignore")

LIME = "#b6ff3c"
RED = "#ff4539"
AMBER = "#ffd166"
BLUE = "#4a9eff"
GREY = "#767676"
OP_STYLE = {"add": "yellow", "heat": "orange3", "regrind": "magenta", "xrd": "cyan", "stop": "green", "invalid": "red"}
RESULTS = Path(__file__).resolve().parent.parent / "results"


def bar(fraction: float, width: int = 22) -> str:
    filled = int(round(fraction * width))
    return "█" * filled + "·" * (width - filled)


def phase_table(fractions: dict[str, float], target: str, charged: set[str]) -> Table:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(ratio=3)
    table.add_column(ratio=4)
    table.add_column(justify="right", ratio=1)
    for formula, fraction in list(fractions.items())[:7]:
        if fraction < 0.005:
            continue
        style = f"bold {LIME}" if formula == target else ("dim" if formula in charged else "white")
        table.add_row(Text(formula, style=style), Text(bar(fraction), style=LIME if formula == target else GREY), Text(f"{fraction:.2f}", style=style))
    if not fractions:
        table.add_row(Text("empty crucible", style="dim"), "", "")
    return table


def describe(record: dict) -> str:
    op = record["op"]
    if op == "add":
        return " + ".join(f"{amount:g} {formula}" for formula, amount in record["precursors"].items())
    if op == "heat":
        return f"{record['temperature_c']:.0f} °C · {record['time_h']:g} h · {record['atmosphere']}"
    if op == "xrd":
        return " ".join(f"{formula} {value:.2f}" for formula, value in list(record["phases"].items())[:4])
    if op == "invalid":
        return record.get("reason", "")
    return ""


class Screen:
    def __init__(self, env: SynthesisGym, agent_name: str):
        self.env = env
        self.agent_name = agent_name
        self.reason = Text("waiting for the agent…", style="dim")
        self.action_json = "{}"
        self.true_w: dict[int, float] = {}
        self.caption: str = ""
        self.title: str | None = None
        self.layout = Layout()
        self.layout.split_column(Layout(name="header", size=5), Layout(name="body", ratio=3), Layout(name="trajectory", ratio=2))
        self.layout["body"].split_row(Layout(name="agent", ratio=5), Layout(name="state", ratio=6))
        self.layout["state"].split_column(Layout(name="hidden"), Layout(name="observed"))

    def header(self) -> Group:
        env = self.env
        line = Text()
        line.append(" SYNTHESIS GYM ", style=f"bold #111111 on {LIME}")
        line.append(f"  target {self.title or env.task.task_id}", style="bold white")
        line.append("  ·  ", style="dim")
        line.append(f"agent {self.agent_name}", style=BLUE)
        line.append("  ·  ", style="dim")
        line.append(f"step {env.steps}/{env.max_steps}", style="white")
        line.append("  ·  ", style="dim")
        line.append(f"cost {env.process_cost:.3f}", style=AMBER)
        line.append("  ·  ", style="dim")
        line.append(f"return {env.total_reward:+.3f}", style=LIME if env.total_reward >= 0 else RED)
        caption = Text(f" {self.caption}", style=f"bold {LIME}") if self.caption else Text("")
        return Group(line, Rule(style="dim"), caption)

    def render(self) -> Layout:
        env = self.env
        self.layout["header"].update(self.header())
        agent_body = Group(self.reason, Text(""), Syntax(self.action_json, "json", theme="ansi_dark", background_color="default"))
        self.layout["agent"].update(Panel(agent_body, title="[bold]agent[/] [dim]reasoning · action[/]", title_align="left", box=box.SQUARE, border_style=GREY))
        truth = env.crucible.weight_fractions()
        hidden = Group(
            phase_table(truth, env.task.target, set(env.charged)),
            Text(""),
            Text.assemble(("contact ", "dim"), (f"{env.crucible.contact:.2f}", "white"), ("   mass loss ", "dim"), (f"{env.crucible.mass_loss:.3f}", "white")),
        )
        self.layout["hidden"].update(Panel(hidden, title="[bold]crucible[/] [dim]true state · hidden from agent[/]", title_align="left", box=box.SQUARE, border_style=GREY))
        if env.last_xrd:
            age = env.steps - env.xrd_step
            observed = Group(phase_table(env.last_xrd, env.task.target, set(env.charged)), Text(f"measured {age} step(s) ago · noise σ 0.03 · <3 wt% hidden", style="dim"))
        else:
            observed = Text("no XRD yet: the agent is blind to the crucible", style="dim")
        self.layout["observed"].update(Panel(observed, title="[bold]xrd[/] [dim]what the agent sees[/]", title_align="left", box=box.SQUARE, border_style=BLUE))
        rows = Table(expand=True, box=box.SIMPLE_HEAD, header_style="dim", padding=(0, 1))
        rows.add_column("#", width=3, justify="right", style="dim")
        rows.add_column("op", width=9)
        rows.add_column("detail", ratio=1)
        rows.add_column("w_target", width=9, justify="right")
        rows.add_column("reward", width=8, justify="right")
        for record in env.history:
            style = OP_STYLE.get(record["op"], "white")
            w_target = self.true_w.get(record["step"])
            rows.add_row(str(record["step"]), Text(f"[{record['op']:<7}]", style=style), describe(record),
                         Text(f"{w_target:.2f}" if w_target is not None else "", style=LIME if (w_target or 0) >= 0.9 else "white"),
                         Text(f"{record['reward']:+.3f}", style=LIME if record["reward"] > 0 else "dim"))
        self.layout["trajectory"].update(Panel(rows, title="[bold]trajectory[/] [dim]true target fraction after each step[/]", title_align="left", box=box.SQUARE, border_style=GREY))
        return self.layout


def make_agent(name: str):
    if name == "claude":
        return ClaudeAgent()
    if name == "claude-retrieval":
        return ClaudeAgent(retrieval=True)
    if name == "random":
        return RandomAgent(0)
    return HeuristicAgent()


def scoreboard(console: Console) -> None:
    path = RESULTS / "metrics.json"
    if not path.exists():
        return
    metrics = json.loads(path.read_text())
    table = Table(title=f"[bold]{metrics['suite']}[/] · held-out targets · Wilson 95% CI", box=box.SIMPLE_HEAD, header_style="dim", title_justify="left")
    table.add_column("agent", style=BLUE)
    for label in ["precursor set = lit.", "set + T ±150 °C", "T MAE °C", "sim w ≥ 0.9", "mean return"]:
        table.add_column(label, justify="right")
    for agent, entry in metrics["agents"].items():
        def cell(key):
            item = entry[key]
            return f"{item['rate']:.0%} [dim][{item['wilson95'][0]:.0%},{item['wilson95'][1]:.0%}][/]"
        table.add_row(agent, cell("precursor_match"), cell("recipe_match"), f"{entry['temp_mae_c']:.0f}", cell("success_sim"), f"{entry['mean_reward']:.2f}")
    console.print(table)
    console.print(f"[dim]median-temperature predictor MAE {metrics['median_temperature_baseline_mae_c']:.0f} °C · literature = published recipes, purity unknown · sim metrics are simulated[/]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="SynthGym-v0/hard/LiBiP2O7")
    parser.add_argument("--agent", default="claude")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    console = Console()
    env = SynthesisGym(task_spec(args.task))
    observation, _ = env.reset(seed=args.seed)
    agent = make_agent(args.agent)
    agent.reset(env.task)
    screen = Screen(env, {"claude": "claude-opus-5.5"}.get(args.agent, args.agent))
    with Live(screen.render(), console=console, screen=True, auto_refresh=False) as live:
        time.sleep(args.delay)
        while True:
            screen.reason = Text("thinking…", style="dim italic")
            live.update(screen.render(), refresh=True)
            action = agent.act(observation)
            screen.reason = Text(getattr(agent, "last_reason", "") or "scripted baseline policy", style="white")
            screen.action_json = json.dumps(action, indent=2)
            live.update(screen.render(), refresh=True)
            time.sleep(args.delay * 0.6)
            observation, reward, terminated, truncated, info = env.step(action)
            screen.true_w[env.steps] = info["w_target"]
            live.update(screen.render(), refresh=True)
            time.sleep(args.delay)
            if terminated or truncated:
                break
        final = info["w_target"]
        verdict = Text.assemble(("  episode done  ", f"bold #111111 on {LIME if final >= 0.9 else RED}"),
                                (f"  true target fraction {final:.3f}  ·  return {env.total_reward:+.3f}  ·  bonus {'yes' if final >= 0.9 else 'no'}", "bold"))
        screen.reason = verdict
        live.update(screen.render(), refresh=True)
        time.sleep(args.delay * 2.5)
    console.print(screen.header())
    scoreboard(console)


if __name__ == "__main__":
    main()
