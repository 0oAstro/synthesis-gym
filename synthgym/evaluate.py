"""Run agents over a task suite and score them.

Headline metrics do not read the simulator: literature match and temperature error against the
held-out published recipes. Simulated metrics (terminal reward, w_target) are reported separately.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from synthgym.env import SynthesisGym
from synthgym.tasks import task_spec

TEMPERATURE_TOLERANCE_C = 150.0


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (centre - half, centre + half)


def mcnemar(first: pd.Series, second: pd.Series) -> float:
    """Exact McNemar p-value for paired booleans."""
    only_first = int((first & ~second).sum())
    only_second = int((~first & second).sum())
    if only_first + only_second == 0:
        return 1.0
    return float(binomtest(only_first, only_first + only_second, 0.5).pvalue)


def run_episode(agent, task_id: str, seed: int = 0, on_step=None) -> dict:
    task = task_spec(task_id)
    env = SynthesisGym(task)
    observation, info = env.reset(seed=seed)
    agent.reset(task)
    first_add = None
    heats = []
    xrd_count = 0
    while True:
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        record = observation["history"][-1]
        if record["op"] == "add" and first_add is None:
            first_add = tuple(sorted(record["precursors"]))
        if record["op"] == "heat":
            heats.append(record["temperature_c"])
        if record["op"] == "xrd":
            xrd_count += 1
        if on_step is not None:
            on_step(env, action, record, info)
        if terminated or truncated:
            break
    literature_sets = {tuple(sorted(recipe.precursors)) for recipe in task.recipes}
    set_match = first_add in literature_sets if first_add else False
    hottest = max(heats) if heats else None
    matching_temps = [recipe.max_temp_c for recipe in task.recipes if tuple(sorted(recipe.precursors)) == first_add]
    published = float(np.median([recipe.max_temp_c for recipe in task.recipes]))
    temp_error = abs(hottest - published) if hottest is not None else None
    full_match = bool(set_match and hottest is not None and any(abs(hottest - t) <= TEMPERATURE_TOLERANCE_C for t in matching_temps))
    return {
        "agent": agent.name,
        "task_id": task_id,
        "tier": task.tier,
        "precursor_match": bool(set_match),
        "recipe_match": full_match,
        "temp_error_c": temp_error,
        "w_target": info["w_target"],
        "success_sim": info["w_target"] >= 0.9,
        "total_reward": info["total_reward"],
        "steps": len(observation["history"]),
        "heats": len(heats),
        "xrd": xrd_count,
        "stopped_without_xrd": xrd_count == 0,
        "invalid": sum(1 for record in observation["history"] if record["op"] == "invalid"),
    }


def run_suite(make_agent, task_ids: list[str], seed: int = 0, workers: int = 8) -> pd.DataFrame:
    """make_agent() must return a fresh agent; one per task so threads never share state."""
    def one(task_id: str) -> dict:
        return run_episode(make_agent(), task_id, seed)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, task_ids))
    return pd.DataFrame(rows)


def median_temperature_baseline(task_ids: list[str]) -> float:
    """MAE of always predicting the training-split median max temperature."""
    from synthgym.tasks import load_world
    median = float(load_world()[0].train_recipes["max_temp_c"].median())
    errors = []
    for task_id in task_ids:
        task = task_spec(task_id)
        errors.append(abs(median - float(np.median([recipe.max_temp_c for recipe in task.recipes]))))
    return float(np.mean(errors))


def summarize(frame: pd.DataFrame) -> dict:
    summary = {}
    for agent, rows in frame.groupby("agent"):
        n = len(rows)
        entry = {"n": n}
        for column in ["precursor_match", "recipe_match", "success_sim"]:
            k = int(rows[column].sum())
            low, high = wilson(k, n)
            entry[column] = {"rate": k / n, "wilson95": [round(low, 3), round(high, 3)]}
        entry["temp_mae_c"] = float(rows["temp_error_c"].dropna().mean())
        entry["mean_w_target"] = float(rows["w_target"].mean())
        entry["mean_reward"] = float(rows["total_reward"].mean())
        entry["audit"] = {
            "mean_heats": float(rows["heats"].mean()),
            "stopped_without_xrd": float(rows["stopped_without_xrd"].mean()),
            "mean_invalid": float(rows["invalid"].mean()),
        }
        entry["by_tier"] = {tier: {"n": len(group), "success_sim": float(group["success_sim"].mean()),
                                   "precursor_match": float(group["precursor_match"].mean())}
                            for tier, group in rows.groupby("tier")}
        summary[agent] = entry
    return summary
