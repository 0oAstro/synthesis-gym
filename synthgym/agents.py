"""Baseline agents. Each agent has reset(task) and act(observation) -> action dict (the JSON form)."""

import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np

from synthgym.kinetics import melt_k
from synthgym.tasks import Knowledge, TaskSpec, cations, load_world

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "claude"
CLAUDE_MODEL = "us.anthropic.claude-opus-5-5"
AWS_REGION = "us-east-1"
AWS_PROFILE = "default"
MAX_CLAUDE_CALLS = int(os.environ.get("MAX_CLAUDE_CALLS", "3000"))


def common_precursors(knowledge: Knowledge, target: str, per_element: int = 1) -> dict[str, list[str]]:
    """Most frequent training precursors per target cation (single-cation compounds only)."""
    choices = {}
    for element in sorted(cations(target)):
        choices[element] = knowledge.precursors_by_element.get(element, [])[:per_element]
    return choices


def tamman_temperature(knowledge: Knowledge, precursors: list[str]) -> float:
    mean_melt = float(np.mean([melt_k(p) for p in precursors]))
    predicted = knowledge.temperature_slope * mean_melt + knowledge.temperature_intercept
    return float(min(max(predicted, 400.0), 1600.0))


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self, task: TaskSpec) -> None:
        knowledge = load_world()[0]
        options = common_precursors(knowledge, task.target, per_element=8)
        self.precursors = [self.rng.choice(choice) for choice in options.values() if choice]
        self.plan = [
            {"op": "add", "precursors": self.precursors},
            {"op": "heat", "temperature_c": self.rng.uniform(400, 1500), "time_h": self.rng.uniform(1, 24),
             "atmosphere": self.rng.choice(["air", "O2", "inert", "reducing", "vacuum"])},
            {"op": "stop"},
        ]

    def act(self, observation: dict) -> dict:
        return self.plan[min(observation["step"], len(self.plan) - 1)]


class HeuristicAgent:
    """Modal training precursor per cation, Tamman-fit temperature, calcine, regrind, sinter, measure, stop."""

    name = "heuristic"

    def reset(self, task: TaskSpec) -> None:
        knowledge = load_world()[0]
        options = common_precursors(knowledge, task.target)
        self.precursors = [choice[0] for choice in options.values() if choice]
        temperature = tamman_temperature(knowledge, self.precursors) if self.precursors else 1000.0
        self.plan = [
            {"op": "add", "precursors": self.precursors},
            {"op": "heat", "temperature_c": max(temperature - 300, 300), "time_h": 6, "atmosphere": "air"},
            {"op": "regrind"},
            {"op": "heat", "temperature_c": temperature, "time_h": 12, "atmosphere": "air"},
            {"op": "xrd"},
            {"op": "stop"},
        ]

    def act(self, observation: dict) -> dict:
        return self.plan[min(observation["step"], len(self.plan) - 1)]


SYSTEM_PROMPT = """You run a solid-state synthesis furnace. Goal: make the target phase as pure as possible, cheaply.
The crucible contents are hidden. You see only what you added, mass loss, and XRD results you pay for.
Each turn, reply with a short reason (one or two sentences) and then exactly one JSON action on its own line:
{"op": "add", "precursors": ["A", "B"]}            env balances moles to the target stoichiometry
{"op": "add", "precursors": {"A": 1.0, "B": 0.5}}   explicit moles (use to top up a missing element)
{"op": "heat", "temperature_c": 900, "time_h": 12, "atmosphere": "air"}   atmosphere: air, O2, inert, reducing, vacuum
{"op": "regrind"}   restores particle contact (reheating without it is slower)
{"op": "xrd"}       measure phase weight fractions (noisy, minor phases hidden)
{"op": "stop"}      end; reward = true target weight fraction (+0.5 if >= 0.9) minus costs
Costs: heat 0.02 to 0.10 (hotter and longer cost more), xrd 0.02, add and regrind 0.01, invalid 0.05.
Only use precursors from the allowed list. You have at most 10 steps including stop."""


def extract_json(text: str) -> dict | None:
    candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        for end in range(len(candidate), 0, -1):
            if candidate[end - 1] != "}":
                continue
            try:
                return json.loads(candidate[:end])
            except json.JSONDecodeError:
                continue
    return None


class ClaudeAgent:
    """Claude Opus 5.5 on Bedrock (personal account, us-east-1), with a disk cache keyed by the prompt."""

    calls = 0

    def __init__(self, retrieval: bool = False, model: str = CLAUDE_MODEL):
        self.retrieval = retrieval
        self.model = model
        self.name = "claude-retrieval" if retrieval else "claude"
        self.fallbacks = 0
        self.last_reason = ""
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import AnthropicBedrock
            self._client = AnthropicBedrock(aws_region=AWS_REGION, aws_profile=AWS_PROFILE)
        return self._client

    def reset(self, task: TaskSpec) -> None:
        self.task = task
        knowledge = load_world()[0]
        names = display_names_cached()
        self.allowed = {element: [names.get(formula, formula) for formula in formulas[:12]]
                        for element, formulas in common_precursors(knowledge, task.target, per_element=12).items()}
        self.fallback = HeuristicAgent()
        self.fallback.reset(task)
        self.messages: list[dict] = []
        context = {"target": task.target, "allowed_precursors_by_element": self.allowed}
        if self.retrieval:
            context["similar_training_recipes"] = retrieve_similar(knowledge, task.target)
        self.intro = json.dumps(context)

    def _call(self, messages: list[dict]) -> str:
        key = hashlib.sha256(json.dumps([self.model, SYSTEM_PROMPT, messages]).encode()).hexdigest()
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["text"]
        if ClaudeAgent.calls >= MAX_CLAUDE_CALLS:
            raise RuntimeError(f"MAX_CLAUDE_CALLS={MAX_CLAUDE_CALLS} reached")
        ClaudeAgent.calls += 1
        response = self.client.messages.create(model=self.model, max_tokens=1200, system=SYSTEM_PROMPT, messages=messages)
        text = "".join(block.text for block in response.content if block.type == "text")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text}))
        return text

    def act(self, observation: dict) -> dict:
        shown = {key: observation[key] for key in ["step", "cost_so_far", "charged", "mass_loss", "last_xrd", "xrd_age", "flags"]}
        if observation["history"]:
            shown["last_step"] = observation["history"][-1]
        content = (self.intro + "\n\n" if not self.messages else "") + "Observation: " + json.dumps(shown)
        self.messages.append({"role": "user", "content": content})
        text = self._call(self.messages)
        action = extract_json(text)
        if action is None or "op" not in action:
            retry = self.messages + [{"role": "assistant", "content": text},
                                     {"role": "user", "content": "Your reply had no valid JSON action. Reply with only the JSON action."}]
            text = self._call(retry)
            action = extract_json(text)
        if action is not None and "op" in action:
            self.messages.append({"role": "assistant", "content": text})
            self.last_reason = text.split("{")[0].strip() or "(json only)"
            return action
        self.fallbacks += 1
        action = self.fallback.act(observation)
        if action["op"] == "stop" and observation["step"] < observation["max_steps"] - 1:
            action = {"op": "xrd"}
        self.messages.append({"role": "assistant", "content": json.dumps(action)})
        self.last_reason = "(fallback to heuristic)"
        return action


_DISPLAY_NAMES: dict[str, str] = {}


def display_names_cached() -> dict[str, str]:
    if not _DISPLAY_NAMES:
        from synthgym.data import display_names
        _DISPLAY_NAMES.update(display_names(load_world()[0].train_recipes))
    return _DISPLAY_NAMES


def retrieve_similar(knowledge: Knowledge, target: str, k: int = 5) -> list[dict]:
    """Nearest training recipes by cation-set overlap (Jaccard); training split only."""
    target_cations = cations(target)
    scored = []
    for row in knowledge.train_recipes.itertuples():
        other = cations(row.target)
        union = target_cations | other
        if not union:
            continue
        score = len(target_cations & other) / len(union)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: -item[0])
    seen = set()
    results = []
    for score, row in scored:
        if row.target in seen:
            continue
        seen.add(row.target)
        results.append({"target": row.target, "precursors": list(row.precursors), "max_temp_c": row.max_temp_c,
                         "total_time_h": row.total_time_h, "atmosphere": row.atmosphere})
        if len(results) == k:
            break
    return results
