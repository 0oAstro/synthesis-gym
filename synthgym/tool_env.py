"""SynthesisGym as a tool-calling environment for TRL GRPO (`environment_factory`).

Each public method is a tool the model can call. Every tool returns the new observation as JSON,
so an XRD result lands in the model's context before its next decision (closed loop).
The episode return from the gym is the reward (`get_reward`).
"""

import json

from synthgym.agents import common_precursors, display_names_cached
from synthgym.env import SynthesisGym
from synthgym.tasks import load_world, task_spec

TOOL_SYSTEM = """You run a solid-state synthesis furnace through tools. Goal: make the target phase as pure as possible, cheaply.
The crucible contents are hidden; you only see what you added, mass loss, and XRD results you pay for.
Typical route: add_precursors, heat, xrd, then regrind and heat again if precursors or intermediates remain, then stop.
Score at stop = true target weight fraction (+0.5 bonus if >= 0.9) minus costs (heat 0.02-0.10, xrd 0.02, add and regrind 0.01, invalid 0.05).
Call exactly one tool per turn. Always finish by calling stop."""


def task_prompt(task_id: str) -> list[dict]:
    task = task_spec(task_id)
    knowledge = load_world()[0]
    names = display_names_cached()
    allowed = {element: [names.get(formula, formula) for formula in formulas]
               for element, formulas in common_precursors(knowledge, task.target, per_element=8).items()}
    user = json.dumps({"target": task.target, "allowed_precursors_by_element": allowed})
    return [{"role": "system", "content": TOOL_SYSTEM}, {"role": "user", "content": user}]


NO_ATTEMPT_REWARD = -0.3


class SynthesisToolEnv:
    def __init__(self):
        self.env: SynthesisGym | None = None
        self.done = False

    def reset(self, task_id: str, **kwargs) -> None:
        self.env = SynthesisGym(task_spec(task_id))
        self.env.reset(seed=0)
        self.done = False
        return None

    def _act(self, action: dict) -> str:
        if self.done:
            return json.dumps({"error": "episode already stopped"})
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated
        shown = {"step": observation["step"], "cost": observation["cost_so_far"], "mass_loss": observation["mass_loss"]}
        last = observation["history"][-1]
        if last["op"] == "invalid":
            shown["invalid"] = last["reason"]
        if last["op"] == "xrd":
            shown["xrd"] = observation["last_xrd"]
        if last["op"] == "add":
            shown["charged"] = observation["charged"]
        if observation["flags"]:
            shown["flags"] = observation["flags"]
        if self.done:
            shown["episode_over"] = True
        return json.dumps(shown)

    def add_precursors(self, precursors: list[str]) -> str:
        """Charge precursors into the crucible; moles are balanced to the target stoichiometry.

        Args:
            precursors: Precursor formulas from the allowed list, e.g. ["BaCO3", "TiO2"].

        Returns:
            The new observation as JSON.
        """
        return self._act({"op": "add", "precursors": list(precursors)})

    def heat(self, temperature_c: float, time_h: float, atmosphere: str) -> str:
        """Heat the crucible.

        Args:
            temperature_c: Temperature in Celsius, 200 to 1700.
            time_h: Hold time in hours, 0.5 to 72.
            atmosphere: One of air, O2, inert, reducing, vacuum.

        Returns:
            The new observation as JSON.
        """
        return self._act({"op": "heat", "temperature_c": temperature_c, "time_h": time_h, "atmosphere": atmosphere})

    def regrind(self) -> str:
        """Regrind the powder to restore particle contact before another heat.

        Returns:
            The new observation as JSON.
        """
        return self._act({"op": "regrind"})

    def xrd(self) -> str:
        """Measure phase weight fractions by X-ray diffraction (noisy; phases under 3 wt% are hidden).

        Returns:
            The new observation as JSON, including last_xrd.
        """
        return self._act({"op": "xrd"})

    def stop(self) -> str:
        """End the episode. The score is the true target weight fraction minus costs.

        Returns:
            The final observation as JSON.
        """
        return self._act({"op": "stop"})

    def get_reward(self) -> float:
        if not self.done:
            self.env.step({"op": "stop"})
            self.done = True
        if not self.env.charged:
            return NO_ATTEMPT_REWARD
        return float(self.env.total_reward)
