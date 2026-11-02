# Synthesis Gym architecture, candidate 1: phase-assemblage MDP

(measured) = `python3 -I` over `data/raw/`, 2026-11-02.

## 1. State model

**Hidden state** `S_t`:
- `phases: dict[str, float]`: weight fraction of every crystalline phase in the crucible (precursors, intermediates, target). Sums to 1 after volatiles leave.
- `contact: float in (0, 1]`: reactant contact factor. Starts at 1.0, multiplied by 0.5 after every heat step (sintered agglomerates), reset to 1.0 by `regrind`.
- `mass_loss: float`: cumulative fraction of initial mass lost as CO2, H2O, NH3, NOx, O2.

**Observation** `O_t`: the target, the action history, `mass_loss` (a balance reading, free), and the phase assemblage **only as of the last `xrd` action**, with its age in steps. Phase fractions from `xrd` carry N(0, 0.03) noise and drop phases under 3 wt%. Between XRDs the agent is blind to phases: partial observability is real and priced.

**Why history matters.** The transition of §2 reacts whatever phases exist. Heating BaO2 + CuO + Y2O3 at 800 C produces BaCuO2 and Y2Cu2O5 before Ba2YCu3O7 (ARROWS3 YBCO: BaCuO2 appears in 66 of 149 verified runs, Y2Cu2O5 in 27, measured). The intermediates have a smaller driving force to the target than the precursors did, so the same 900 C step reaches less target from a sample that went through 800 C than from a fresh charge. `contact` makes "reheat without regrind" worse than "regrind then reheat". Both effects depend on the path, not on the last action. Multi-step protocols are the norm: 15,579 of 31,782 Ceder records have two or more heating steps, 10,829 of them raise T, median gap 300 C (measured).

**Alternatives considered.**
1. History-in-observation only (SPEC v1). Rejected: transition depends on (target, action) only, so it is a contextual bandit; the ML critique is accepted by the owner.
2. Microstructure state (particle size, diffusion lengths, nucleation). Rejected: no data on disk calibrates it; ARROWS3 reports phases and fractions, nothing finer.
3. Learned latent state from recipe sequences (RNN over Ceder operations). Rejected: the 15k multi-step recipes carry no intermediate labels, so the latent would be fit to the final success flag only, the bandit again in disguise.
Phase assemblage is the one state the real data observes, so sim and furnace share a state type.

## 2. Transition model

One `heat(T, t, atm)` step, applied to `S_t`:

**Physics (formation energies).**
1. Candidate phases `C`: every `mp_nostruct` formula whose elements are a subset of the target chemsys plus C, H, N, with `e_hull <= 0.10 eV/atom`, plus the precursor formulas, plus volatiles {CO2, H2O, NH3, NO2, O2}. Every ARROWS3 product phase checked is present (measured); Y2(CO3)3 and NH4H2PO4 are not, so such precursors take `e_form` from a 40-row hand table in `thermo.py` and raise the `missing_thermo` flag.
2. Pairwise reactions, the ARROWS3 convention: for every pair of present phases (and each single phase, for decompositions), solve the element balance `A x = b` with `scipy.optimize.nnls` over products in `C`, and keep the product set with the most negative `dG = dE_form - n_gas * 2.1e-3 eV/K * T_K`, both per reactant atom. The entropy term is only the gas release; solids are 0 K. `dG >= -0.02 eV/atom` means no reaction.
3. Atmosphere gate: a product containing a reducible cation (Cu, Ni, Co, Fe3+, Pb, Bi, Ag, Mn4+, V5+, Mo6+, W6+) is blocked under `reducing`; a product that releases O2 is blocked under `O2`; nothing else depends on atmosphere. (Assumption; the chemist's list.)

**Learned (fitted scalars, not a black box).** Reaction extent in time `t` hours:

```
x = 1 - exp(-t * contact / tau),   tau = tau0 * exp(-beta * (T_K / T_on - 1)) * exp(gamma * dG)
T_on = alpha_class * min(Tm_K of the two reactants)
```

- `alpha_class` per anion class of the reactants (oxide, carbonate, nitrate, hydroxide, phosphate, other): fitted on the **training split of Ceder only** as the 10th percentile of `first_heating_T_K / min Tm_K` over recipes containing that class. Compound Tm from a 100-row table (covers 87.7 % of precursor occurrences, chemist point 1); no elemental proxies.
- `tau0, beta, gamma`: three scalars fitted by `scipy.optimize.minimize` on ARROWS3, minimising mean L1 error between simulated and observed weight fractions after one heat step at the file's protocol (LTOPO `Common Experimental Conditions`; YBCO and NTMO assume 4 h in air, recorded as an assumption). Fit set: LTOPO (58 verified runs) plus 24 of the 47 YBCO precursor sets. Report set: the other 23 YBCO sets and NTMO (18 runs). The fitted values and the held-out L1 are written to `results/kinetics_fit.json`.

After the reaction with largest `x * |dG|` is applied (one reaction per step; the step is sequential over `n_sub = 4` sub-steps of `t/4` so that intermediates formed early can react later within the same hold), `contact *= 0.5`, `mass_loss` accumulates released gas mass.

**Calibration against A-Lab.** The A-Lab CSV has recipes for 35 Success, 7 Partial and 0 of 15 Failed targets (every recipe is 4 h). The sim runs the recorded recipe for Success and Partial, and the heuristic recipe of §7 for Failed. Reported as "target feasibility AUC", target level, Success vs Failed binary, Partial as the middle class of an ordinal AUC.

**What is assumed, listed.** Pairwise reactions only; 0 K solid energies; a single gas-entropy constant; `contact` halving; XRD noise 0.03; one reaction per sub-step. Each is a named constant in `synthgym/constants.py`.

## 3. Spaces

```python
V = len(train_vocab)      # precursors seen in training split only
P = len(phase_vocab)      # candidate phases across all tasks, built from train chemsys
ATM = ["air", "O2", "Ar", "N2", "reducing", "vacuum"]   # "unspecified" maps to air

action_space = gym.spaces.Dict({
    "op":          Discrete(5),            # 0 charge, 1 heat, 2 regrind, 3 xrd, 4 stop
    "precursors":  MultiBinary(V),         # read only when op == charge
    "temperature": Box(200.0, 1700.0, (1,)),
    "log_time":    Box(np.log(0.5), np.log(72.0), (1,)),
    "atmosphere":  Discrete(len(ATM)),
})
observation_space = gym.spaces.Dict({
    "target_frac":  Box(0, 1, (103,)),     # element fractions of the target
    "step":         Discrete(9),
    "phase_ids":    MultiDiscrete([P + 1] * 8),   # last XRD, 0 = empty slot
    "phase_fracs":  Box(0, 1, (8,)),
    "xrd_age":      Discrete(9),           # 8 = never measured
    "mass_loss":    Box(0, 1, (1,)),
    "history":      Box(-1, 1, (8, 10)),   # per step: op one-hot(5), T/1700, log t scaled, atm/5, cost, reward
})
```

LLM JSON form, both directions, validated by one function `parse_action(dict) -> Action | Invalid(reason)`:

```json
{"op": "charge", "precursors": ["BaCO3", "Y2O3", "CuO"]}
{"op": "heat", "temperature_c": 900, "time_h": 12, "atmosphere": "air"}
{"op": "regrind"}  {"op": "xrd"}  {"op": "stop"}
```

Observation to the LLM: `{"target": "YBa2Cu3O7", "step": 3, "last_xrd": {"BaCuO2": 0.31, "Y2O3": 0.12, "CuO": 0.2, "YBa2Cu3O7": 0.37}, "xrd_age": 0, "mass_loss": 0.18, "history": [...], "flags": ["BaCO3 below reaction onset 700 C", "Cu2+ product blocked under reducing"]}`. `flags` come from the gate and onset checks of §2, never from a learned model.

Validity (invalid: reward -0.2, state unchanged, step counted): `charge` only at step 0; every non-volatile target element supplied; no precursor brings a non-volatile element absent from the target; other ops require a prior `charge`.

## 4. Reward and termination

Let `f_T` be the true target weight fraction in `S_t` (the env knows it; the agent sees it only through `xrd`).

```
r_heat    = -(0.02 + 0.08 * (T_c / 1700) * (log1p(t_h) / log1p(72)))     # 900 C x 12 h = -0.054
r_regrind = -0.01     r_xrd = -0.02     r_invalid = -0.2
process cost is capped at -0.30 per episode
r_terminal (on stop or step 8) = f_T + 0.5 * [f_T >= 0.9]
```

Deterministic by default: `x` is the mean extent. `stochastic=True` samples `x ~ Beta` around it; never the default. "Stop immediately" scores 0; a 30-minute anneal yields `x` near 0 at any realistic `tau`, so it scores about -0.03; a two-step calcine-regrind-sinter route to `f_T = 0.95` scores about +1.3. Terminates on `stop`; truncates at 8 steps. `info` carries `true_phases`, `flags`, `cost_so_far`, `validity`.

## 5. Task suite and registry

A task is `(target_formula, allowed_vocab, max_steps, tier)`. Targets come from the cleaned Ceder table after the chemsys split: 85/15 by sorted element set, seed 0, with every training chemsys that is a subset or superset of any test chemsys moved out of training, and Ba-Cu-O-Y, Li-O-P-Ti, Mo-Na-O-Te and all A-Lab chemsys removed from training.

Tiers, assigned from data, not opinion:
- `easy`: binary or ternary oxide target, all precursors oxides, literature `T <= 1000 C`.
- `medium`: carbonate, nitrate or phosphate precursor needed, or quaternary target.
- `hard`: ARROWS3 shows a competing intermediate within 0.05 eV/atom of the target, or the target has a reducible cation (atmosphere matters), or the literature route has 3+ heating steps.

Ids: `SynthGym-v0/easy/BaTiO3`, `SynthGym-v0/hard/YBa2Cu3O7`, and suites `SynthGym-v0/test-200` (200 held-out targets, fixed order, stratified by tier), `SynthGym-v0/real/arrows3`, `SynthGym-v0/real/alab`. `v0` freezes split seed, vocab hash and kinetics constants in `synthgym/registry/v0.json`.

```python
def make(task_id: str, backend: Literal["sim", "lab"] = "sim", **kw) -> SynthesisEnv
def list_tasks(suite: str) -> list[str]
def task_spec(task_id: str) -> TaskSpec
```

## 6. Real-lab tier

`SynthesisEnv` takes a `Backend` with one method: `apply(state: LabState, action: Action) -> LabState`. `SimBackend` is §2. `LabBackend.apply` writes `lab/queue/<episode>/<step>.json`, blocks on `lab/results/<episode>/<step>.json` (polling, timeout configurable), and reads back the same `phases` dict from a Rietveld or reference-intensity-ratio refinement. `xrd` on the lab backend is the real measurement; between XRDs the lab observation is identical to the sim's. Because state, action and observation types are shared, an agent's policy runs unchanged.

Sim-to-real gap per task: replay the lab episode's actions through `SimBackend` from the same charge and report `gap = mean over XRD steps of L1(phases_sim, phases_lab)` plus `|f_T_sim - f_T_lab|` at the end, written to `results/gap/<task_id>.json`. The ARROWS3 held-out L1 of §2 is the first value of this metric, before any furnace is plugged in.

## 7. Evaluation protocol

Held-out targets: `N = 200` from `SynthGym-v0/test-200`. All agents run the same targets, same seed.

Primary, independent of the sim:
- **Literature match**: exact precursor set at the first `charge` (top-1), and final `heat` temperature within 150 C of the literature maximum. Paired McNemar between agents; Wilson 95 % intervals. Reported with retrieval and without.
- **Temperature MAE** against the literature maximum, with the median predictor (247 C MAE, ML critique) as the floor baseline.
- **Real-outcome AUC**: the sim's predicted `f_T` after the recorded protocol against observed `>= 50 wt%` on ARROWS3, overall and in the 900 C stratum (28 positive vs 10 negative), per source and macro-averaged; A-Lab target feasibility AUC as in §2. The 900 C stratum cannot be temperature in disguise.

Secondary, labelled "simulated": mean terminal reward and `f_T >= 0.9` rate per agent and per tier.

Baselines, all re-run on this split: `Random` (valid random charge, uniform T, t, atm, 2 heats); `Heuristic` (modal training precursor per element, T from the Tamman fit on training data, 12 h, one calcine at T-300 C then regrind then sinter, atmosphere from the reducible-cation rule); `Retrieval` (nearest training recipe by Magpie cosine with subset/superset chemsys excluded, replayed step by step); `Claude` (Opus 5.5 on Bedrock, JSON, flags fed back, retry once then heuristic fallback, count logged; disk cache `cache/claude/<sha256>.json`; `MAX_CLAUDE_CALLS=2000`, about 200 targets x 8 steps).

## 8. Module layout

```
synthgym/
  data.py        load_ceder(path) -> pd.DataFrame; load_mp(path) -> dict[str, MPRow]
                 load_arrows3(path) -> list[RealRun]; load_alab(path) -> list[RealTarget]
                 split_chemsys(df, seed) -> Split(train_idx, test_idx, removed_chemsys)
  formula.py     reduce(formula) -> str; elements(formula) -> dict[str, float]; is_volatile(formula) -> bool
  thermo.py      candidates(chemsys) -> list[Phase]; balance(reactants, products) -> np.ndarray | None
                 reaction_dG(reactants, products, T_K) -> float
  kinetics.py    @dataclass Kinetics(tau0, beta, gamma, alpha: dict[str, float])
                 extent(k, dG, T_K, t_h, contact, reactants) -> float
                 fit_kinetics(runs: list[RealRun], sim: SimBackend) -> Kinetics
  state.py       @dataclass LabState(phases: dict[str, float], contact: float, mass_loss: float)
                 @dataclass Action(op, precursors, temperature_c, time_h, atmosphere)
  backend.py     class Backend(Protocol): apply(state, action) -> tuple[LabState, list[str]]  # flags
                 class SimBackend, class LabBackend(queue_dir)
  env.py         class SynthesisEnv(gym.Env): reset(seed, options) / step(action)
                 parse_action(dict) -> Action | Invalid;  observe(state, memory) -> dict
  registry.py    make, list_tasks, task_spec; v0.json
  agents/        random.py, heuristic.py, retrieval.py, claude.py   # each: act(obs_json) -> dict
  evaluate.py    run_suite(agent, suite, seed) -> pd.DataFrame; real_outcome_auc(sim) -> dict
scripts/         fetch_data.py, fit_kinetics.py, run_eval.py, demo.py
results/         kinetics_fit.json, metrics.json, real_outcome_roc.png, literature_match.png
```

`SimBackend.apply` is under 80 lines of plain loops.

## 9. v0 cut (about 5 hours)

| Hour | Ships |
|---|---|
| 1 | `data.py` from the loader snippet, chemsys split with subset/superset exclusion, train vocab, Tm table, `formula.reduce`. |
| 2 | `thermo.py`: candidates, nnls balance, `dG`; `kinetics.extent` with hand-set `(tau0=4, beta=6, gamma=-8, alpha=0.67)`. |
| 3 | `SimBackend`, `SynthesisEnv`, `parse_action`, flags; `fit_kinetics` on LTOPO + YBCO half, `results/kinetics_fit.json` with held-out L1. |
| 4 | Random, Heuristic, Claude agents; `real_outcome_auc` on ARROWS3 (stratified) and A-Lab; `real_outcome_roc.png`. |
| 5 | `run_eval` on `test-200` for Random, Heuristic, Claude plain; `metrics.json`, `literature_match.png`; `demo.py` replaying cached results. |

Deferred to v1: Retrieval agent and the with/without report; `stochastic=True`; `LabBackend` beyond the queue stub and the gap metric; tiers beyond easy/medium/hard thresholds listed above; ordinal A-Lab AUC. Deferred to v2: a learned extent model (LightGBM on `(dG, T/T_on, class, atm)`) trained only on real rows ingested from `lab/results/`, which is the one retraining loop the owner allows; MACE-MP-0 energies for phases missing from `mp_nostruct`; particle-contact state richer than one scalar.

## 10. Risks and how the design shows them

- **Kinetics are three scalars fit on 207 runs of two chemsys.** Visible: `kinetics_fit.json` prints held-out L1 per source; NTMO is never fit on. If the held-out L1 exceeds 0.25 the README says the sim is uncalibrated outside YBCO/LTOPO.
- **Pairwise, most-negative-dG selection misses kinetically favoured minority paths.** Visible: for every ARROWS3 run the sim logs phases observed but never produced; `results/missed_phases.json` names them.
- **Formation-energy coverage.** 35.5 % of Ceder rows have full `mp_nostruct` coverage for precursors and target; missing phases fall back to the hand table or are excluded from `C`. Visible: `missing_thermo` flag rate per tier in `metrics.json`; tasks with any missing energy are tagged `thermo_incomplete` and reported separately.
- **The `contact` factor is uncalibrated.** It is the one path-dependence mechanism with no data behind it. Visible: an ablation run with `contact` frozen at 1 is in `metrics.json`; if agent rankings do not change, the README says regrind is cosmetic in v0.
- **Literature positives mean "published", not phase-pure.** Visible: every caption says so; chemsys capped at 20 training rows for Tamman and vocab statistics.
- **The 900 C stratum has 38 runs.** A Wilson interval on an AUC of 0.75 at n = 38 is roughly 0.58 to 0.88; printed in the title of the ROC.

## Rationale

Chosen: an explicit phase-assemblage state driven by formation-energy pairwise reactions with a three-parameter extent law fitted on ARROWS3, because it is the only state the real data measures, it makes every step path-dependent through intermediates and contact, and it needs no pseudo-negatives at all; the real-outcome AUC is then a property of the simulator, not of a classifier trained on labels the founder invented. Rejected: the SPEC v1 LightGBM oracle on perturbation negatives (bandit, and four of its negative types were measured as wrong or trivial); a learned latent-state model (no intermediate labels to fit it); a microstructure simulator (no data to calibrate); Bernoulli reward (variance only); the env/judge oracle pair (replicas of one bias, and the headline no longer depends on either).

State: document complete, untested against code. Next step: hour 1 of §9.
