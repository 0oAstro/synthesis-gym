"""Evaluate agents on SynthGym-v0/test-200 and write results/.

Usage: uv run python scripts/run_eval.py random heuristic claude claude-retrieval
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthgym.agents import ClaudeAgent, HeuristicAgent, RandomAgent
from synthgym.evaluate import mcnemar, median_temperature_baseline, run_suite, summarize
from synthgym.tasks import list_tasks

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent.parent / "results"

AGENTS = {
    "random": (lambda: RandomAgent(0), 8),
    "heuristic": (HeuristicAgent, 8),
    "claude": (lambda: ClaudeAgent(retrieval=False), 8),
    "claude-retrieval": (lambda: ClaudeAgent(retrieval=True), 8),
}


def plot(summary: dict, path: Path) -> None:
    agents = list(summary)
    metrics = [("precursor_match", "precursor set = literature"), ("recipe_match", "set + T within 150 C"),
               ("success_sim", "simulated w_target >= 0.9")]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (key, label) in zip(axes, metrics):
        rates = [summary[a][key]["rate"] for a in agents]
        lows = [summary[a][key]["rate"] - summary[a][key]["wilson95"][0] for a in agents]
        highs = [summary[a][key]["wilson95"][1] - summary[a][key]["rate"] for a in agents]
        axis.bar(agents, rates, yerr=[lows, highs], color="#b6ff3c", edgecolor="#111111", capsize=6)
        axis.set_title(label, fontsize=15)
        axis.set_ylim(0, 1)
        axis.tick_params(labelsize=12)
    n = summary[agents[0]]["n"]
    figure.suptitle(f"SynthGym-v0 test-200 (n={n} held-out targets, Wilson 95% CI). Literature = published, purity unknown.", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def main(names: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    task_ids = list_tasks()
    frames = []
    for name in names:
        factory, workers = AGENTS[name]
        frame = run_suite(factory, task_ids, workers=workers)
        frame.to_csv(RESULTS / f"episodes_{name}.csv", index=False)
        frames.append(frame)
        print(f"{name}: done", flush=True)
    all_frames = []
    for path in sorted(RESULTS.glob("episodes_*.csv")):
        all_frames.append(pd.read_csv(path))
    combined = pd.concat(all_frames)
    summary = summarize(combined)
    agents = list(summary)
    paired = {}
    for first in agents:
        for second in agents:
            if first < second:
                a = combined[combined.agent == first].set_index("task_id").sort_index()
                b = combined[combined.agent == second].set_index("task_id").sort_index()
                common = a.index.intersection(b.index)
                paired[f"{first} vs {second}"] = {
                    column: mcnemar(a.loc[common, column].astype(bool), b.loc[common, column].astype(bool))
                    for column in ["precursor_match", "recipe_match", "success_sim"]
                }
    metrics = {"suite": "SynthGym-v0/test-200", "agents": summary, "mcnemar_p": paired,
               "median_temperature_baseline_mae_c": median_temperature_baseline(task_ids),
               "claude_api_calls_this_run": ClaudeAgent.calls}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot(summary, RESULTS / "success_by_agent.png")
    print(json.dumps({a: {k: v for k, v in e.items() if k != "by_tier"} for a, e in summary.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["random", "heuristic"])
