"""Load the Ceder text-mined solid-state recipes into a clean table.

Source: Kononova et al., Sci. Data 2019, release 2020-07-13
(github.com/CederGroupHub/text-mined-synthesis_public, file solid-state_dataset_20200713.json.xz).
"""

import json
import lzma
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CEDER_FILE = RAW_DIR / "solid-state_dataset_20200713.json.xz"

ATMOSPHERES = ["air", "O2", "inert", "reducing", "vacuum", "unspecified"]

GAS_WORDS = {
    "air": "air",
    "o2": "O2",
    "oxygen": "O2",
    "flowing oxygen": "O2",
    "ar": "inert",
    "argon": "inert",
    "n2": "inert",
    "nitrogen": "inert",
    "he": "inert",
    "helium": "inert",
    "vacuum": "vacuum",
}

TIME_TO_HOURS = {
    "h": 1.0, "hour": 1.0, "hours": 1.0, "hr": 1.0, "hrs": 1.0,
    "min": 1 / 60, "minutes": 1 / 60, "minute": 1 / 60, "mins": 1 / 60,
    "s": 1 / 3600, "sec": 1 / 3600,
    "d": 24.0, "day": 24.0, "days": 24.0,
    "week": 168.0, "weeks": 168.0,
}


@dataclass(frozen=True)
class HeatStep:
    temperature_c: float
    time_h: float | None
    atmosphere: str


def normalize_formula(formula: str) -> str | None:
    """Return the pymatgen reduced formula, or None when the string does not parse."""
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def material_to_formula(material: dict) -> str | None:
    """Build a reduced formula from the dataset's own per-material element dicts.

    Hydrates arrive as several composition parts (Al(NO3)3 x1 + H2O x9); we sum them.
    Returns None for variable stoichiometry (amounts that are not numbers).
    """
    totals: dict[str, float] = {}
    for part in material["composition"]:
        try:
            part_amount = float(part["amount"])
        except ValueError:
            return None
        for element, amount_text in part["elements"].items():
            try:
                amount = float(amount_text)
            except ValueError:
                return None
            totals[element] = totals.get(element, 0.0) + amount * part_amount
    if not totals:
        return None
    try:
        return Composition(totals).reduced_formula
    except Exception:
        return None


def chemsys(formula: str) -> str:
    """Sorted element set joined by '-', e.g. 'Ba-O-Ti'."""
    return "-".join(sorted(element.symbol for element in Composition(formula).elements))


def classify_atmosphere(words: list[str]) -> str | None:
    labels = set()
    for word in words:
        text = word.strip().lower()
        if "h2" in text or "hydrogen" in text or "forming gas" in text or text == "co":
            labels.add("reducing")
        elif text in GAS_WORDS:
            labels.add(GAS_WORDS[text])
    for preferred in ["reducing", "vacuum", "inert", "O2", "air"]:
        if preferred in labels:
            return preferred
    return None


def _peak(entry: dict) -> float | None:
    numbers = [value for value in (entry.get("values") or []) if value is not None]
    if entry.get("max_value") is not None:
        numbers.append(entry["max_value"])
    if not numbers:
        return None
    return max(numbers)


def heat_steps(record: dict) -> list[HeatStep]:
    """One HeatStep per HeatingOperation that names a temperature, in recipe order."""
    steps = []
    last_atmosphere = "unspecified"
    for operation in record["operations"]:
        if operation["type"] != "HeatingOperation":
            continue
        conditions = operation["conditions"]
        temperatures = []
        for entry in conditions.get("heating_temperature") or []:
            peak = _peak(entry)
            if peak is None:
                continue
            if entry["units"] == "K":
                peak -= 273.15
            temperatures.append(peak)
        if not temperatures:
            continue
        time_h = None
        time_entries = conditions.get("heating_time") or []
        if time_entries:
            peak = _peak(time_entries[0])
            factor = TIME_TO_HOURS.get(time_entries[0]["units"])
            if peak is not None and factor is not None:
                time_h = peak * factor
        gas = classify_atmosphere(conditions.get("heating_atmosphere") or [])
        if gas is not None:
            last_atmosphere = gas
        steps.append(HeatStep(max(temperatures), time_h, last_atmosphere))
    return steps


def record_to_row(record: dict) -> dict | None:
    target = material_to_formula(record["target"])
    if target is None:
        return None
    display_by_formula = {}
    for material in record["precursors"]:
        formula = material_to_formula(material)
        if formula is None:
            return None
        display_by_formula[formula] = material["material_formula"]
    precursors = sorted(display_by_formula)
    if not precursors:
        return None
    steps = heat_steps(record)
    known_times = [step.time_h for step in steps if step.time_h is not None]
    return {
        "target": target,
        "chemsys": chemsys(target),
        "precursors": tuple(precursors),
        "precursor_names": tuple(display_by_formula[formula] for formula in precursors),
        "steps": tuple(steps),
        "n_steps": len(steps),
        "max_temp_c": max((step.temperature_c for step in steps), default=None),
        "total_time_h": sum(known_times) if known_times else None,
        "atmosphere": steps[-1].atmosphere if steps else "unspecified",
        "reaction_str": record["reaction_string"],
        "doi": record["doi"],
    }


def load_recipes(path: Path = CEDER_FILE) -> pd.DataFrame:
    """Parse, normalise and clean the Ceder dataset into one row per recipe."""
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["reactions"]
    rows = []
    for record in records:
        row = record_to_row(record)
        if row is not None:
            rows.append(row)
    table = pd.DataFrame(rows)
    keep = table["max_temp_c"].isna() | table["max_temp_c"].between(100, 2000)
    keep &= table["total_time_h"].isna() | table["total_time_h"].between(0.05, 500)
    table = table[keep].drop_duplicates(subset=["target", "precursors", "max_temp_c", "doi"])
    return table.reset_index(drop=True)


def display_names(table: pd.DataFrame) -> dict[str, str]:
    """Most common as-written name for each reduced formula (NH4H2PO4 rather than PH6NO4)."""
    counts: dict[str, dict[str, int]] = {}
    for formulas, names in zip(table["precursors"], table["precursor_names"]):
        for formula, name in zip(formulas, names):
            per_name = counts.setdefault(formula, {})
            per_name[name] = per_name.get(name, 0) + 1
    return {formula: max(per_name, key=per_name.get) for formula, per_name in counts.items()}
