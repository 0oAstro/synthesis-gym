"""Tasks, the leak-safe split, and the registry.

Split: by chemical system (sorted element set of the target), 85/15, seed 0. A training chemsys that
is a subset or superset of any test chemsys leaves training, and so does every chemsys in the
ARROWS3 and A-Lab real-outcome sets. Vocabulary, precursor frequencies and the temperature fit come
from the training split only.
"""

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from synthgym.data import load_recipes
from synthgym.kinetics import melt_k
from synthgym.realdata import load_alab_outcomes, load_arrows_runs
from synthgym.thermo import formation_table

VERSION = "SynthGym-v0"
TASKS_FILE = Path(__file__).resolve().parent.parent / "tasks" / "v0.json"
VOLATILE_ELEMENTS = {"O", "H", "C", "N"}
REDUCIBLE_CATIONS = {"Cu", "Ni", "Co", "Fe", "Pb", "Bi", "Ag", "Mn", "V", "Mo", "W"}
MIN_VOCAB_COUNT = 3
TEST_TARGETS = 400
SUITE_SIZE = 200


@dataclass(frozen=True)
class LiteratureRecipe:
    precursors: tuple[str, ...]
    max_temp_c: float
    total_time_h: float | None
    atmosphere: str


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    target: str
    chemsys: str
    tier: str
    split: str
    published_temp_c: float
    published_atmosphere: str
    recipes: tuple[LiteratureRecipe, ...]


def cations(formula: str) -> set[str]:
    return {str(element) for element in Composition(formula).elements} - VOLATILE_ELEMENTS


def _elements(chemsys: str) -> frozenset[str]:
    return frozenset(chemsys.split("-"))


def split_chemsys(recipes: pd.DataFrame, seed: int = 0) -> tuple[set[str], set[str]]:
    """Test chemsys are sampled only among systems whose targets have a formation energy (the sim
    needs one), until TEST_TARGETS targets are covered. Everything else is train, minus near
    duplicates of test systems and the real-outcome systems."""
    energies = formation_table()
    eligible = sorted({chemsys for chemsys, target in zip(recipes["chemsys"], recipes["target"]) if target in energies})
    random.Random(seed).shuffle(eligible)
    targets_by_chemsys = recipes.groupby("chemsys")["target"].apply(lambda column: {t for t in column if t in energies})
    test: set[str] = set()
    covered = 0
    for chemsys in eligible:
        if covered >= TEST_TARGETS:
            break
        test.add(chemsys)
        covered += len(targets_by_chemsys[chemsys])
    test_sets = [_elements(chemsys) for chemsys in test]
    real_sets = set()
    for run in load_arrows_runs():
        real_sets.add(frozenset(str(e) for e in Composition(run.target).elements))
    for outcome in load_alab_outcomes():
        real_sets.add(frozenset(str(e) for e in Composition(outcome.target).elements))
    train = set()
    for chemsys in recipes["chemsys"].unique():
        if chemsys in test:
            continue
        elements = _elements(chemsys)
        if elements in real_sets:
            continue
        if any(_near_duplicate(elements, other) for other in test_sets):
            continue
        train.add(chemsys)
    return train, test


def _near_duplicate(first: frozenset[str], second: frozenset[str]) -> bool:
    """True for a subset/superset pair that differs by one element, both ternary or larger.

    Catches doped variants (Ba-O-Ti vs Ba-O-Ti-Zr) without letting a binary test chemsys such as
    Li-O remove every lithium ternary from training.
    """
    if len(first) < 3 or len(second) < 3:
        return False
    if not (first <= second or second <= first):
        return False
    return len(first ^ second) == 1


@dataclass
class Knowledge:
    """Everything an agent may know: built from the training split only."""

    vocab: list[str]
    precursors_by_element: dict[str, list[str]]
    temperature_slope: float
    temperature_intercept: float
    train_recipes: pd.DataFrame


def build_knowledge(train_recipes: pd.DataFrame) -> Knowledge:
    counts = Counter(p for precursors in train_recipes["precursors"] for p in precursors)
    vocab = sorted(p for p, count in counts.items() if count >= MIN_VOCAB_COUNT)
    by_element: dict[str, Counter] = {}
    for precursors in train_recipes["precursors"]:
        for precursor in precursors:
            if precursor not in counts or counts[precursor] < MIN_VOCAB_COUNT:
                continue
            precursor_cations = cations(precursor)
            if len(precursor_cations) != 1:
                continue
            element = next(iter(precursor_cations))
            by_element.setdefault(element, Counter())[precursor] += 1
    precursors_by_element = {element: [p for p, _ in counter.most_common()] for element, counter in by_element.items()}
    with_temp = train_recipes.dropna(subset=["max_temp_c"])
    mean_melt = np.array([np.mean([melt_k(p) for p in precursors]) for precursors in with_temp["precursors"]])
    slope, intercept = np.polyfit(mean_melt, with_temp["max_temp_c"].to_numpy(), 1)
    return Knowledge(vocab, precursors_by_element, float(slope), float(intercept), train_recipes)


def tier_of(target: str, recipes: list[LiteratureRecipe], n_steps: int) -> str:
    target_cations = cations(target)
    if target_cations & REDUCIBLE_CATIONS or len(target_cations) >= 4:
        return "hard"
    oxide_only = all(all(set(map(str, Composition(p).elements)) - {"O"} for p in r.precursors) and
                     all(not (set(map(str, Composition(p).elements)) & {"C", "N", "H"}) for p in r.precursors) for r in recipes)
    if len(target_cations) <= 2 and oxide_only and n_steps <= 1:
        return "easy"
    return "medium"


def _task_from_group(target: str, group: pd.DataFrame, split: str) -> TaskSpec:
    recipes = tuple(
        LiteratureRecipe(tuple(row.precursors), float(row.max_temp_c), None if pd.isna(row.total_time_h) else float(row.total_time_h), row.atmosphere)
        for row in group.itertuples()
    )
    named = [r.atmosphere for r in recipes if r.atmosphere != "unspecified"]
    atmosphere = Counter(named).most_common(1)[0][0] if named else "air"
    tier = tier_of(target, list(recipes), int(group["n_steps"].median()))
    return TaskSpec(
        task_id=f"{VERSION}/{tier}/{target}",
        target=target,
        chemsys=group["chemsys"].iloc[0],
        tier=tier,
        split=split,
        published_temp_c=float(group["max_temp_c"].median()),
        published_atmosphere=atmosphere,
        recipes=recipes,
    )


@lru_cache(maxsize=1)
def load_world(seed: int = 0) -> tuple[Knowledge, dict[str, TaskSpec], dict[str, list[str]]]:
    """Returns (training knowledge, all tasks by id, suites by name)."""
    recipes = load_recipes().dropna(subset=["max_temp_c"])
    train_chemsys, test_chemsys = split_chemsys(recipes, seed)
    knowledge = build_knowledge(recipes[recipes["chemsys"].isin(train_chemsys)])
    energies = formation_table()
    vocab = set(knowledge.vocab)
    tasks: dict[str, TaskSpec] = {}
    test_ids = []
    for target, group in recipes[recipes["chemsys"].isin(test_chemsys)].groupby("target"):
        if target not in energies:
            continue
        if not any(set(precursors) <= vocab for precursors in group["precursors"]):
            continue
        task = _task_from_group(target, group, "test")
        tasks[task.task_id] = task
        test_ids.append(task.task_id)
    train_ids = []
    for target, group in recipes[recipes["chemsys"].isin(train_chemsys)].groupby("target"):
        if target not in energies:
            continue
        if not any(set(precursors) <= vocab for precursors in group["precursors"]):
            continue
        task = _task_from_group(target, group, "train")
        if task.task_id in tasks:
            continue
        tasks[task.task_id] = task
        train_ids.append(task.task_id)
    rng = random.Random(seed)
    by_tier: dict[str, list[str]] = {}
    for task_id in sorted(test_ids):
        by_tier.setdefault(tasks[task_id].tier, []).append(task_id)
    suite = []
    tiers = sorted(by_tier)
    per_tier = SUITE_SIZE // len(tiers)
    for tier in tiers:
        chosen = by_tier[tier][:]
        rng.shuffle(chosen)
        suite.extend(chosen[:per_tier])
    leftovers = [task_id for task_id in sorted(test_ids) if task_id not in set(suite)]
    rng.shuffle(leftovers)
    suite.extend(leftovers[: SUITE_SIZE - len(suite)])
    suites = {f"{VERSION}/test-{SUITE_SIZE}": sorted(suite), f"{VERSION}/test-all": sorted(test_ids), f"{VERSION}/train": sorted(train_ids)}
    return knowledge, tasks, suites


def list_tasks(suite: str = f"{VERSION}/test-{SUITE_SIZE}") -> list[str]:
    return load_world()[2][suite]


def task_spec(task_id: str) -> TaskSpec:
    return load_world()[1][task_id]


def write_manifest(path: Path = TASKS_FILE) -> dict:
    knowledge, tasks, suites = load_world()
    manifest = {
        "version": VERSION,
        "split_seed": 0,
        "vocab_size": len(knowledge.vocab),
        "suites": suites,
        "tier_counts": dict(Counter(tasks[t].tier for t in suites[f"{VERSION}/test-{SUITE_SIZE}"])),
        "n_train_recipes": len(knowledge.train_recipes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest
