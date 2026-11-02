"""SynthesisGym: the Gymnasium environment.

The agent works on a hidden crucible: add precursors, heat, regrind, pay for XRD, stop.
The terminal reward is the true target weight fraction (+0.5 bonus at >= 0.9) minus bounded costs.
Agents speak JSON (see parse_action); `step` also accepts the same dict.
"""

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from pymatgen.core import Composition

from synthgym.kinetics import Kinetics
from synthgym.sim import ATMOSPHERES, Crucible, stabilizing_shift, xrd
from synthgym.tasks import VOLATILE_ELEMENTS, TaskSpec, cations, load_world, task_spec
from synthgym.thermo import balance

OPS = ["add", "heat", "regrind", "xrd", "stop"]
MAX_STEPS = 10
COST_ADD = 0.01
COST_REGRIND = 0.01
COST_XRD = 0.02
COST_INVALID = 0.05
PROCESS_COST_CAP = 0.40
SUCCESS_THRESHOLD = 0.9
SUCCESS_BONUS = 0.5


def heat_cost(temperature_c: float, time_h: float) -> float:
    return 0.02 + 0.08 * (temperature_c / 1700.0) * math.log1p(time_h) / math.log1p(72.0)


@dataclass
class Action:
    op: str
    precursors: dict[str, float] = field(default_factory=dict)
    temperature_c: float = 0.0
    time_h: float = 0.0
    atmosphere: str = "air"


@dataclass
class Invalid:
    reason: str


def parse_action(raw: dict, target: str, vocab: set[str], aliases: dict[str, str] | None = None) -> Action | Invalid:
    """Validate an agent action. A precursor list (not dict) is balanced to the target by the env."""
    op = raw.get("op")
    if op not in OPS:
        return Invalid(f"unknown op {op!r}; use one of {OPS}")
    if op == "add":
        given = raw.get("precursors")
        if isinstance(given, dict):
            names = list(given)
        elif isinstance(given, list):
            names = list(given)
        else:
            return Invalid("add needs 'precursors' as a list or a {formula: moles} dict")
        reduced = {}
        for name in names:
            if aliases and name in aliases:
                formula = aliases[name]
            else:
                try:
                    formula = Composition(name).reduced_formula
                except Exception:
                    return Invalid(f"cannot parse formula {name!r}")
            if formula not in vocab:
                return Invalid(f"{name} is not in the precursor vocabulary")
            reduced[formula] = name
        target_cations = cations(target)
        for formula in reduced:
            extra = cations(formula) - target_cations
            if extra:
                return Invalid(f"{formula} brings {sorted(extra)}, which the target does not contain")
            if target_cations and target_cations <= cations(formula):
                return Invalid(f"{formula} already contains every cation of the target")
        if isinstance(given, dict):
            amounts = {}
            for formula, name in reduced.items():
                try:
                    amounts[formula] = float(given[name])
                except (TypeError, ValueError):
                    return Invalid(f"amount for {name} must be a number")
                if not 0 < amounts[formula] <= 20:
                    return Invalid(f"amount for {name} must be in (0, 20] mol")
        else:
            coefficients = balance(list(reduced), [target])
            if coefficients is None:
                return Invalid(f"{sorted(reduced)} cannot be balanced to {target}")
            amounts = {formula: float(coefficients.get(formula, 0.0)) for formula in reduced if coefficients.get(formula, 0.0) > 1e-9}
        return Action("add", precursors=amounts)
    if op == "heat":
        try:
            temperature = float(raw["temperature_c"])
            time_h = float(raw["time_h"])
        except (KeyError, TypeError, ValueError):
            return Invalid("heat needs numeric temperature_c and time_h")
        atmosphere = raw.get("atmosphere", "air")
        if atmosphere not in ATMOSPHERES:
            return Invalid(f"atmosphere must be one of {ATMOSPHERES}")
        if not 200 <= temperature <= 1700:
            return Invalid("temperature_c must be in [200, 1700]")
        if not 0.5 <= time_h <= 72:
            return Invalid("time_h must be in [0.5, 72]")
        return Action("heat", temperature_c=temperature, time_h=time_h, atmosphere=atmosphere)
    return Action(op)


class SynthesisGym(gym.Env):
    """One episode synthesises one target. Observations are JSON-ready dicts."""

    metadata = {"render_modes": []}

    def __init__(self, task: TaskSpec | str, kinetics: Kinetics | None = None, max_steps: int = MAX_STEPS, xrd_noise: float = 0.03):
        self.task = task_spec(task) if isinstance(task, str) else task
        self.knowledge = load_world()[0]
        self.vocab = set(self.knowledge.vocab)
        from synthgym.agents import display_names_cached
        self.aliases = {name: formula for formula, name in display_names_cached().items()}
        self.kinetics = kinetics or Kinetics.load()
        self.max_steps = max_steps
        self.xrd_noise = xrd_noise
        self.target_shift = stabilizing_shift(self.task.target, self.task.published_temp_c, self.task.published_atmosphere)
        self.action_space = gym.spaces.Dict({
            "op": gym.spaces.Discrete(len(OPS)),
            "precursors": gym.spaces.MultiBinary(len(self.knowledge.vocab)),
            "temperature_c": gym.spaces.Box(200.0, 1700.0, (1,), np.float32),
            "time_h": gym.spaces.Box(0.5, 72.0, (1,), np.float32),
            "atmosphere": gym.spaces.Discrete(len(ATMOSPHERES)),
        })
        self.observation_space = gym.spaces.Dict({
            "step": gym.spaces.Discrete(max_steps + 1),
            "cost_so_far": gym.spaces.Box(0.0, 10.0, (1,), np.float32),
            "mass_loss": gym.spaces.Box(-1.0, 1.0, (1,), np.float32),
        })

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.crucible = Crucible(target=self.task.target, target_shift=self.target_shift)
        self.steps = 0
        self.process_cost = 0.0
        self.total_reward = 0.0
        self.history: list[dict] = []
        self.last_xrd: dict[str, float] | None = None
        self.xrd_step: int | None = None
        self.charged: dict[str, float] = {}
        return self.observe(), self.info()

    def observe(self) -> dict:
        return {
            "target": self.task.target,
            "step": self.steps,
            "max_steps": self.max_steps,
            "cost_so_far": round(self.process_cost, 4),
            "charged": {formula: round(amount, 3) for formula, amount in self.charged.items()},
            "mass_loss": round(self.crucible.mass_loss, 4),
            "last_xrd": self.last_xrd,
            "xrd_age": None if self.xrd_step is None else self.steps - self.xrd_step,
            "history": self.history[-self.max_steps:],
            "flags": self.flags(),
        }

    def flags(self) -> list[str]:
        """Hints computed only from what the agent observed (charge and last XRD), never hidden state."""
        notes = []
        if self.last_xrd:
            leftovers = [formula for formula in self.last_xrd if formula in self.charged]
            if leftovers:
                notes.append(f"unreacted precursors seen by XRD: {', '.join(leftovers)}")
            target_fraction = self.last_xrd.get(self.task.target, 0.0)
            if target_fraction == 0.0:
                notes.append("target not detected by XRD")
            elif target_fraction < SUCCESS_THRESHOLD:
                others = [f for f in self.last_xrd if f != self.task.target and f not in self.charged]
                if others:
                    notes.append(f"intermediates present: {', '.join(others)}")
        return notes

    def info(self) -> dict:
        return {
            "true_phases": self.crucible.weight_fractions() if hasattr(self, "crucible") else {},
            "w_target": self.crucible.weight_fractions().get(self.task.target, 0.0) if hasattr(self, "crucible") else 0.0,
            "contact": getattr(getattr(self, "crucible", None), "contact", 1.0),
            "task_id": self.task.task_id,
        }

    def _charge_cost(self, cost: float) -> float:
        allowed = max(0.0, PROCESS_COST_CAP - self.process_cost)
        charged = min(cost, allowed)
        self.process_cost += charged
        return -charged

    def step(self, action: dict | Action):
        parsed = action if isinstance(action, Action) else parse_action(action, self.task.target, self.vocab, self.aliases)
        self.steps += 1
        reward = 0.0
        terminated = False
        record: dict = {"step": self.steps}
        if isinstance(parsed, Invalid):
            reward = -COST_INVALID
            record.update(op="invalid", reason=parsed.reason)
        elif parsed.op == "add":
            self.crucible.add(parsed.precursors)
            for formula, amount in parsed.precursors.items():
                self.charged[formula] = self.charged.get(formula, 0.0) + amount
            reward = self._charge_cost(COST_ADD)
            record.update(op="add", precursors={f: round(a, 3) for f, a in parsed.precursors.items()})
        elif not self.charged and parsed.op in ("heat", "regrind", "xrd"):
            reward = -COST_INVALID
            record.update(op="invalid", reason=f"{parsed.op} before any add")
        elif parsed.op == "heat":
            self.crucible.heat(parsed.temperature_c, parsed.time_h, parsed.atmosphere, self.kinetics)
            reward = self._charge_cost(heat_cost(parsed.temperature_c, parsed.time_h))
            record.update(op="heat", temperature_c=parsed.temperature_c, time_h=parsed.time_h, atmosphere=parsed.atmosphere)
        elif parsed.op == "regrind":
            self.crucible.regrind()
            reward = self._charge_cost(COST_REGRIND)
            record.update(op="regrind")
        elif parsed.op == "xrd":
            self.last_xrd = xrd(self.crucible, self.np_random, self.xrd_noise)
            self.xrd_step = self.steps
            reward = self._charge_cost(COST_XRD)
            record.update(op="xrd", phases=self.last_xrd)
        elif parsed.op == "stop":
            terminated = True
            record.update(op="stop")
        truncated = not terminated and self.steps >= self.max_steps
        info = self.info()
        if terminated or truncated:
            w_target = info["w_target"]
            reward += w_target + (SUCCESS_BONUS if w_target >= SUCCESS_THRESHOLD else 0.0)
            info["terminal_w_target"] = w_target
        record["reward"] = round(reward, 4)
        self.history.append(record)
        self.total_reward += reward
        info["total_reward"] = self.total_reward
        return self.observe(), reward, terminated, truncated, info


def make(task_id: str, **kwargs) -> SynthesisGym:
    return SynthesisGym(task_id, **kwargs)


gym.register(id="SynthGym-v0", entry_point="synthgym.env:SynthesisGym")
