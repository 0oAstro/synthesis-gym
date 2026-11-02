"""Real synthesis outcomes, including failures, from two public sources.

ARROWS3 (Szymanski et al., Nat. Commun. 2023, github.com/njszym/ARROWS, Examples/*/Exp.json):
per precursor set and temperature, the XRD-identified product phases and weight fractions.
Only entries marked "Experimentally Verified" are real; the rest are model predictions and are dropped.

A-Lab (Szymanski et al., Nature 2023, SI file 41586_2023_6734_MOESM3_ESM.zip): one row per target
with Success / Partial / Failed. Failed rows carry no recipe, so A-Lab supports target-level metrics only.
"""

import csv
import json
import re
from dataclasses import dataclass

from synthgym.data import RAW_DIR, normalize_formula

ARROWS_TARGETS = {
    "YBCO": "Ba2YCu3O7",
    "NTMO": "Na2Te3Mo3O16",
    "LTOPO": "LiTiPO5",
}


@dataclass(frozen=True)
class RealRun:
    source: str
    target: str
    precursors: tuple[str, ...]
    temperature_c: float
    time_h: float
    atmosphere: str
    phases: tuple[tuple[str, float], ...]
    target_fraction: float


def phase_formula(label: str) -> str:
    """'Na2Te3Mo3O16_5.cif' -> reduced formula of 'Na2Te3Mo3O16' (the suffix is a space group number)."""
    name = label.removesuffix(".cif").rsplit("_", 1)[0]
    return normalize_formula(name) or name


def load_arrows_runs() -> list[RealRun]:
    runs = []
    for system, target_name in ARROWS_TARGETS.items():
        with open(RAW_DIR / f"{system}_Exp.json") as handle:
            universal = json.load(handle)["Universal File"]
        target = normalize_formula(target_name)
        for precursor_key, entry in universal.items():
            if "Temperatures" not in entry:
                continue
            precursors = tuple(sorted(normalize_formula(name.strip()) or name.strip() for name in precursor_key.split(",")))
            for temperature_text, result in entry["Temperatures"].items():
                if not result.get("Experimentally Verified"):
                    continue
                phases = tuple(
                    (phase_formula(label), float(fraction))
                    for label, fraction in zip(result["products"], result["product weight fractions"])
                )
                target_fraction = sum(fraction for formula, fraction in phases if formula == target) / 100.0
                runs.append(RealRun(
                    source=f"arrows3-{system}",
                    target=target,
                    precursors=precursors,
                    temperature_c=float(temperature_text.split()[0]),
                    time_h=4.0,
                    atmosphere="air",
                    phases=phases,
                    target_fraction=target_fraction,
                ))
    return runs


@dataclass(frozen=True)
class ALabOutcome:
    target: str
    result: str
    precursors: tuple[str, ...]
    temperature_c: float | None
    time_h: float | None


def load_alab_outcomes() -> list[ALabOutcome]:
    outcomes = []
    with open(RAW_DIR / "alab_synthesis_results.csv", newline="", encoding="latin-1") as handle:
        for row in csv.DictReader(handle):
            target = normalize_formula(row["Target"])
            if target is None:
                continue
            result = "Success" if row["Result"].startswith("Success") else row["Result"]
            precursors = []
            for part in row["Precursors + Masses"].split("+"):
                words = part.split()
                if words:
                    precursors.append(normalize_formula(words[-1]) or words[-1])
            temperatures = [float(number) for number in re.findall(r"\d+\.?\d*", row["Temperature (C)"])]
            durations = [float(number) for number in re.findall(r"\d+\.?\d*", row["Duration (hours)"])]
            outcomes.append(ALabOutcome(
                target=target,
                result=result,
                precursors=tuple(sorted(precursors)),
                temperature_c=max(temperatures) if temperatures else None,
                time_h=sum(durations) if durations else None,
            ))
    return outcomes
