"""Fit the extent law (tau0, beta, gamma, alpha) to ARROWS3 phase weight fractions.

Fit set: LTOPO and half of the YBCO precursor sets. Held out: the other YBCO sets and all of NTMO.
Writes models/kinetics.json and results/kinetics_fit.json.
"""

import json
import random
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from synthgym.data import load_recipes
from synthgym.kinetics import Kinetics
from synthgym.realdata import RealRun, load_arrows_runs
from synthgym.sim import Crucible, stabilizing_shift
from synthgym.thermo import balance

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).resolve().parent.parent / "results"
DEFAULT_PUBLISHED_C = 1000.0


def charge_amounts(run: RealRun) -> dict[str, float]:
    if len(set(run.amounts)) > 1:
        return dict(zip(run.precursors, run.amounts))
    coefficients = balance(list(run.precursors), [run.target])
    if coefficients is None:
        return dict(zip(run.precursors, run.amounts))
    return {formula: coefficients.get(formula, 0.0) for formula in run.precursors if coefficients.get(formula, 0.0) > 0}


def simulate_run(run: RealRun, kinetics: Kinetics, shift: float) -> dict[str, float]:
    crucible = Crucible(target=run.target, target_shift=shift)
    crucible.add(charge_amounts(run))
    crucible.heat(run.temperature_c, run.time_h, run.atmosphere, kinetics)
    return crucible.weight_fractions()


def l1_error(predicted: dict[str, float], run: RealRun) -> float:
    observed = {formula: fraction / 100.0 for formula, fraction in run.phases}
    phases = set(predicted) | set(observed)
    return sum(abs(predicted.get(phase, 0.0) - observed.get(phase, 0.0)) for phase in phases)


SHIFTS: dict[str, float] = {}
RUNS: list[RealRun] = []


def _evaluate(args: tuple[int, tuple[float, float, float, float]]) -> tuple[float, float]:
    index, params = args
    run = RUNS[index]
    kinetics = Kinetics(*params)
    predicted = simulate_run(run, kinetics, SHIFTS[run.target])
    return l1_error(predicted, run), predicted.get(run.target, 0.0)


def evaluate_all(pool: Pool, indices: list[int], params) -> tuple[np.ndarray, np.ndarray]:
    results = pool.map(_evaluate, [(index, tuple(params)) for index in indices])
    return np.array([r[0] for r in results]), np.array([r[1] for r in results])


def published_temperatures() -> dict[str, float]:
    recipes = load_recipes()
    medians = recipes.dropna(subset=["max_temp_c"]).groupby("target")["max_temp_c"].median()
    return medians.to_dict()


def auc_or_none(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def report(runs: list[RealRun], errors: np.ndarray, target_pred: np.ndarray) -> dict:
    labels = np.array([run.target_fraction >= 0.5 for run in runs])
    rows = {"n": len(runs), "mean_l1": float(errors.mean()), "auc_target_ge_50": auc_or_none(labels, target_pred)}
    by_temperature = {}
    for temperature in sorted({run.temperature_c for run in runs}):
        mask = np.array([run.temperature_c == temperature for run in runs])
        by_temperature[str(int(temperature))] = {
            "n": int(mask.sum()),
            "positives": int(labels[mask].sum()),
            "mean_l1": float(errors[mask].mean()),
            "auc": auc_or_none(labels[mask], target_pred[mask]),
            "mean_pred_target": float(target_pred[mask].mean()),
            "mean_obs_target": float(np.mean([runs[i].target_fraction for i in np.where(mask)[0]])),
        }
    rows["by_temperature_c"] = by_temperature
    return rows


def main() -> None:
    global RUNS, SHIFTS
    RUNS = load_arrows_runs()
    medians = published_temperatures()
    for target in {run.target for run in RUNS}:
        SHIFTS[target] = stabilizing_shift(target, medians.get(target, DEFAULT_PUBLISHED_C), "air")
    ybco_sets = sorted({run.precursor_set for run in RUNS if run.source == "arrows3-YBCO"})
    random.Random(0).shuffle(ybco_sets)
    fit_ybco_sets = set(ybco_sets[: len(ybco_sets) // 2])
    fit_indices = [i for i, run in enumerate(RUNS) if run.source == "arrows3-LTOPO" or run.precursor_set in fit_ybco_sets]
    held_indices = [i for i in range(len(RUNS)) if i not in set(fit_indices)]
    print(f"fit runs {len(fit_indices)}, held-out runs {len(held_indices)}")

    prior = Kinetics()
    start = np.array([np.log(prior.tau0), prior.beta, prior.gamma, prior.alpha])
    with Pool(8) as pool:
        def loss(vector: np.ndarray) -> float:
            params = (float(np.exp(vector[0])), float(vector[1]), float(vector[2]), float(np.clip(vector[3], 0.2, 1.5)))
            errors, _ = evaluate_all(pool, fit_indices, params)
            value = float(errors.mean())
            print(f"  loss {value:.4f} params {tuple(round(p, 3) for p in params)}", flush=True)
            return value

        prior_params = (prior.tau0, prior.beta, prior.gamma, prior.alpha)
        prior_held, prior_held_pred = evaluate_all(pool, held_indices, prior_params)
        result = minimize(loss, start, method="Nelder-Mead", options={"maxfev": 120, "xatol": 0.02, "fatol": 0.002})
        vector = result.x
        fitted = Kinetics(float(np.exp(vector[0])), float(vector[1]), float(vector[2]), float(np.clip(vector[3], 0.2, 1.5)))
        params = (fitted.tau0, fitted.beta, fitted.gamma, fitted.alpha)
        fit_errors, fit_pred = evaluate_all(pool, fit_indices, params)
        held_errors, held_pred = evaluate_all(pool, held_indices, params)

    fitted.save()
    held_runs = [RUNS[i] for i in held_indices]
    summary = {
        "params": params,
        "fit_set": report([RUNS[i] for i in fit_indices], fit_errors, fit_pred),
        "held_out": {
            source: report(
                [run for run in held_runs if run.source == source],
                held_errors[[run.source == source for run in held_runs]],
                held_pred[[run.source == source for run in held_runs]],
            )
            for source in sorted({run.source for run in held_runs})
        },
        "held_out_prior_mean_l1": float(prior_held.mean()),
        "held_out_fitted_mean_l1": float(held_errors.mean()),
        "target_shift_ev_per_fu": SHIFTS,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "kinetics_fit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
