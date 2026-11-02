"""Plan mode: the model writes a whole recipe (a JSON list of actions) in one reply; the gym runs it.

Used for GRPO training of open models and for their before/after evaluation. It is open-loop: the
model cannot react to XRD mid-episode. The multi-turn agent loop (agents.ClaudeAgent) is closed-loop.
"""

import json

from synthgym.agents import display_names_cached, extract_json, common_precursors
from synthgym.env import SynthesisGym
from synthgym.tasks import TaskSpec, load_world

PLAN_SYSTEM = """You plan solid-state syntheses. Reply with a short reason, then a JSON object {"plan": [...]} listing actions in order.
Actions:
{"op": "add", "precursors": ["A", "B"]}   moles are balanced to the target automatically
{"op": "heat", "temperature_c": 900, "time_h": 12, "atmosphere": "air"}   atmosphere: air, O2, inert, reducing, vacuum; T 200-1700; time 0.5-72
{"op": "regrind"}   restores particle contact before another heat
{"op": "xrd"}       measure (costs 0.02; optional)
The plan ends with stop automatically. At most 9 actions.
Score = true target weight fraction (+0.5 bonus if >= 0.9) minus costs (heat 0.02 to 0.10 each, regrind 0.01, invalid 0.05).
Use only the allowed precursors."""


def plan_prompt(task: TaskSpec) -> list[dict]:
    knowledge = load_world()[0]
    names = display_names_cached()
    allowed = {element: [names.get(formula, formula) for formula in formulas]
               for element, formulas in common_precursors(knowledge, task.target, per_element=8).items()}
    user = json.dumps({"target": task.target, "allowed_precursors_by_element": allowed})
    return [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": user}]


def parse_plan(text: str) -> list[dict] | None:
    parsed = extract_json(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("plan"), list):
        actions = [action for action in parsed["plan"] if isinstance(action, dict)]
        return actions[:9]
    return None


def score_plan(task: TaskSpec, text: str, seed: int = 0) -> dict:
    """Run a plan through the gym. Unparseable text scores -0.5."""
    actions = parse_plan(text)
    if actions is None:
        return {"reward": -0.5, "w_target": 0.0, "parsed": False, "first_add": None, "hottest": None}
    env = SynthesisGym(task)
    env.reset(seed=seed)
    info = {"w_target": 0.0, "total_reward": 0.0}
    first_add = None
    heats = []
    for action in actions:
        if action.get("op") == "stop":
            break
        _, _, terminated, truncated, info = env.step(action)
        record = env.history[-1]
        if record["op"] == "add" and first_add is None:
            first_add = tuple(sorted(record["precursors"]))
        if record["op"] == "heat":
            heats.append(record["temperature_c"])
        if terminated or truncated:
            break
    if not (env.history and (env.steps >= env.max_steps)):
        _, _, _, _, info = env.step({"op": "stop"})
    return {"reward": info["total_reward"], "w_target": info["w_target"], "parsed": True,
            "first_add": first_add, "hottest": max(heats) if heats else None}


class PlanAgent:
    """Wraps a pre-generated plan text so the standard evaluator can score it."""

    def __init__(self, name: str, plans: dict[str, str]):
        self.name = name
        self.plans = plans

    def reset(self, task: TaskSpec) -> None:
        actions = parse_plan(self.plans.get(task.task_id, "")) or []
        self.queue = [action for action in actions if action.get("op") != "stop"][:9] + [{"op": "stop"}]

    def act(self, observation: dict) -> dict:
        return self.queue[min(observation["step"], len(self.queue) - 1)]
