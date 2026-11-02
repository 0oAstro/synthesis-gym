# Synthesis Gym architecture, candidate 3: a crucible with a hidden phase assemblage

Numbers marked (measured) come from `python3 -I` runs over `data/raw/` on 2026-11-02.

## 1. State model

**Hidden state** `S = (A, c, h)`:
- `A`: the phase assemblage, a dict `{formula: moles}` over a per-task phase library `L` (every
  mp_nostruct formula inside the target chemsys with `e_hull <= 0.05 eV/atom`, plus the charged
  precursors and the volatiles CO2, H2O, NH3, O2, NO2). It starts empty and is filled by `add` actions.
- `c in [0,1]`: contact factor. 1.0 after charging or `regrind`; multiplied by 0.5 after each heat.
  It scales the extent of every reaction, so a reheat without regrinding stalls (assumption, §2).
- `h`: thermal history `[(T, t, atm)]`, for cost accounting and logging.

**Observation** `O`: the target, the step index, the cumulative cost, the list of charged
precursors (the agent knows what it put in), the mass-loss fraction since charging (free, exact
from volatiles released), and the **last XRD result** if the agent has paid for one: phases
with weight fraction `>= 0.05` after Gaussian noise `sigma = 0.03`, phases below that dropped.
The assemblage itself is never shown. Stopping without an XRD is legal.

**Why history matters.** ARROWS3 (measured): the same YBCO precursor set gives 0% target at
600 and 700 C in every verified run, partial `Ba2YCu3O7` plus `BaCO3` and `BaCuO2` at 800 C, and
`>= 50%` target in 28 of 38 runs at 900 C. A 800 C step converts the charge into intermediates
(`BaCuO2`, `Y2Cu2O5`), and the step after it reacts those intermediates, whose reaction to the
target has a smaller `|dE|` than the original precursors (driving force consumed). The gate in §2
depends on `dE` and on `c`, so the transition from a 800 C-first episode and from a fresh charge
differ at the same action: heat 900 C. A late `add` reacts with the crucible contents, not the original charge. Ceder agrees:
19,070 of 31,782 raw recipes (60%) have two or more heating operations and 14,907 (47%) contain
a heat, grind or shape, heat sequence (measured).

**Alternatives rejected.** (a) History in the observation only, oracle on (target, action):
the contextual bandit of SPEC v1, closed by the owner. (b) A latent learned state vector
(RNN over actions): nothing in the data labels it, and it cannot be compared to an ARROWS3
product list, so the sim-to-real gap would be undefined. (c) A kinetic model with Avrami
exponents and diffusion lengths: no data on disk carries particle sizes or time-resolved
conversion. The phase-assemblage state is the one whose every component has a counterpart in
`Exp.json` (`products`, `product weight fractions`).

## 2. Transition model

`step(S, a) -> S'` for `a = heat(T, t, atm)`:

1. **Enumerate reactions (physics).** For every pair of solid phases `(p, q)` in `A` with
   `moles > 1e-4`, and each pair alone, propose products from `L`: every subset of at most two
   library phases plus volatiles that balances the pair's elements, balanced with
   `pymatgen.analysis.reaction_calculator.Reaction`. Driving force
   `dE = sum(n_i * e_form_i) over products - sum(n_j * e_form_j) over reactants`, in eV per atom
   of reactants, from `mp_nostruct_20181018` (`e_form`, lowest-energy polymorph per reduced
   formula; 56,750 formulas; covers 35 of the 37 ARROWS3 phases, measured). Keep reactions with
   `dE < -0.01`. 0 K, no entropy: `BaCO3 + TiO2 -> BaTiO3 + CO2` carries the oxide formation
   energy, so the standalone decomposition temperature is never used (chemist point 2).
   Assumption: no solid solutions, melts or metastable products.
2. **Gate each reaction (learned).** `g = P(reaction completes within the step)` from a LightGBM
   bootstrap ensemble (8 members, mean and std) on features:
   `T_c, log10(t_h), atm one-hot without unspecified, dE, T_K / mean Tm_K of reactants
   (compound Tm table for the top-100 precursors, 87.7% of occurrences), carbonate_in_reactants,
   nitrate_or_hydroxide_in_reactants, reducible_cation_present, c, Magpie(target)`.
   Training data: the 15,145 clean Ceder rows with T and time, each turned into reactions by
   running step 1 on its precursor set and labelling every `dE < 0` reaction toward the recorded
   target as 1 at `(T_max, t_sum, atm)`. Pseudo-negatives, balance-preserving only (chemist
   point 4, owner decision): T shifted by -200 to -300 C; time divided by 20; nitrate or hydroxide
   precursor above 1200 C; carbonate precursor below the reaction onsets BaCO3 700, SrCO3 750,
   CaCO3 650, Li2CO3 450, Na2CO3 500 C; reducing atmosphere only for targets with an H2-reducible
   cation (Cu, Ni, Co, Fe3+, Pb, Bi, Ag, Mn4+, V5+, Mo6+, W6+) and only from positives with a
   named atmosphere. No melt flag, no standalone carbonate negative, no element-breaking swaps.
   Positives are "published, purity unknown"; each chemsys is capped at 20 training rows.
3. **Apply (assumption).** Extent `x = c * g`. Process reactions in order of most negative `dE`;
   each consumes `x` of its limiting reactant and adds products; repeat up to 3 passes or until no
   gated reaction moves more than `1e-3` mol (the ARROWS3 pairwise scheme). Then `c <- 0.5 c`.
   Volatile moles leave the crucible and raise mass loss.

`regrind`: `c <- 1`. `add(p, n)`: `A[p] += n`, `c <- 1`. `xrd`: fills the observation. `stop`: terminal.

**Calibration (not training).** No ARROWS3 or A-Lab row, and no Ceder row in their chemsys
(Ba-Cu-O-Y 87 rows, Li-O-P-Ti 18, Mo-Na-O-Te, all A-Lab chemsys), enters training. Replays:
- ARROWS3: for each precursor set and T, run `add` for each precursor at the file's ratios,
  `heat(T, 4 h, air)` (LTOPO files state heating rate and atmosphere under
  `Common Experimental Conditions`), compare predicted weight fractions with `products` and
  `product weight fractions`. Metrics: assemblage L1 distance `sum |w_pred - w_obs|` over `L`,
  and ROC-AUC of predicted target fraction against the label `w_target >= 50%`, stratified by T
  (YBCO verified: 600 C 25, 700 C 36, 800 C 39, 900 C 38, 1000 C 11; NTMO 18; LTOPO 58, measured;
  LTOPO is 56 of 58 positive and is reported as calibration only). The 900 C stratum (28 vs 10) is
  the recipe-level AUC. Formulas are matched by `Composition.reduced_formula`; `Y2(CO3)3` and
  `BaYCu3O7` lack `e_form` and come from a sourced hand table in `thermo.py`.
- A-Lab: masses to moles, `heat(T, 4 h, air)`, AUC of predicted target fraction against Success
  (incl. offline) = 1, Failed = 0, Partial excluded from the binary AUC and included in an
  ordinal AUC (Failed < Partial < Success). Name: target-feasibility AUC, 35 + 15 targets.

## 3. Spaces

Gymnasium, `V` = training-fold precursor vocabulary (1,689 total before fold restriction),
`|L| <= 64` per task (padded), `K = 8` XRD slots, `E = 89` elements.

```python
action_space = spaces.Dict({
    "op": spaces.Discrete(5),                       # 0 add, 1 regrind, 2 heat, 3 xrd, 4 stop
    "precursor": spaces.Discrete(V),                # used by add
    "amount_mol": spaces.Box(0.05, 10.0, (1,)),     # used by add
    "temperature_c": spaces.Box(200.0, 1700.0, (1,)),
    "time_h": spaces.Box(0.5, 72.0, (1,)),
    "atmosphere": spaces.Discrete(6),               # air O2 Ar N2 reducing vacuum
})
observation_space = spaces.Dict({
    "target_elements": spaces.Box(0, 1, (E,)),      # normalised composition
    "step": spaces.Discrete(13),
    "cost_so_far": spaces.Box(0, 1, (1,)),
    "charged": spaces.Box(0, 10, (V,)),             # moles per vocab precursor
    "mass_loss_frac": spaces.Box(0, 1, (1,)),
    "xrd_phase": spaces.MultiDiscrete([65] * K),    # 0 = empty slot, else index into L
    "xrd_frac": spaces.Box(0, 1, (K,)),
    "xrd_age": spaces.Discrete(13),                 # steps since the XRD, 12 = none
})
```

LLM form (`env.to_json(obs)` and `env.parse_json(text)`):
```json
{"target": "Ba2YCu3O7", "step": 3, "cost_so_far": 0.09,
 "charged": {"BaCO3": 2, "Y2O3": 0.5, "CuO": 3}, "mass_loss_frac": 0.11,
 "xrd": {"age": 0, "phases": {"BaCuO2": 0.41, "Y2Cu2O5": 0.22, "BaCO3": 0.18, "Ba2YCu3O7": 0.19}},
 "flags": ["BaCO3 remains: below reaction onset 700 C or time too short"]}
{"op": "heat", "temperature_c": 920, "time_h": 12, "atmosphere": "O2"}
```
`flags` describe the last observed XRD only, never the hidden state.

## 4. Reward and termination

Terminal reward at `stop` or truncation (`max_steps = 12`): `r_T = w_target`, the true solid
weight fraction of the target in `A`. Per-step costs: `add`, `regrind` 0.01; `xrd` 0.03;
`heat` `0.02 + 0.10 * (T_c / 1500) * log(1 + t_h) / log(73)`, so 1100 C x 12 h costs 0.085
and 1700 C x 72 h costs 0.133; the summed heat cost is capped at 0.40 per episode. Total cost is
below 0.6, so any path reaching `w_target >= 0.6` beats stopping at step 0 (reward 0), and a
0.5 h anneal costs 0.045 but leaves `g` low, so `w_target` stays near zero. Invalid actions (foreign element, `heat` before any `add`, out-of-range values) cost 0.05 and
leave `S` unchanged. No Bernoulli sampling; `stochastic=True` samples `g` from the
ensemble members instead of the mean. The episode never terminates on success: the agent decides when to stop, which is what makes
`xrd` worth its price.

## 5. Task suite and registry

A task is `(target, phase library, tier, split)`. Targets are clean fixed-stoichiometry Ceder targets whose
library is fully covered by `e_form`. Split by chemsys, 85/15, seed 0,
with every chemsys that is a subset or superset of a test chemsys removed from training, and all
ARROWS3 and A-Lab chemsys removed. Tiers: 1, two cations, oxide precursors suffice; 2, three
cations or a carbonate, nitrate or phosphate precursor; 3, four or more cations, or a reducible
cation, or `L` contains an intermediate whose reaction to the target has `|dE| < 0.03 eV/atom`
(driving-force trap). Ids: `SynthGym-v0/sim/tier2/BaTiO3`, `SynthGym-v0/arrows3/YBCO/900C`,
`SynthGym-v0/alab/MgCuP2O7`, `SynthGym-v0/lab/<formula>`.

```python
def list_tasks(split: str = "test", tier: int | None = None, source: str = "sim") -> list[TaskSpec]
def make(task_id: str, seed: int = 0, backend: Backend | None = None) -> SynthesisGym
```
`gymnasium.register(id="SynthGym-v0", entry_point="synthgym.env:SynthesisGym")`.
`tasks/v0.json` pins the target lists; a new list is a new version.

## 6. Real-lab tier

```python
class Backend(Protocol):
    def reset(self, task: TaskSpec, seed: int) -> None: ...
    def apply(self, action: Action) -> Observation: ...   # heat, add, regrind, xrd, stop
```
`SimBackend` runs §2. `LabQueueBackend.apply` writes `lab/queue/<episode_id>/<step>.json` with the
action and returns `Observation(pending=True)`; a human runs the furnace and diffractometer and drops `lab/results/<episode_id>/<step>.json` with `{"phases": {...}}`; `apply` polls that file.
Sim-to-real gap per task: replay the real
episode's actions through `SimBackend` and report `gap_L1 = mean over XRD steps of sum |w_sim - w_lab|`
and `gap_target = |w_target_sim - w_target_lab|` at the final step. ARROWS3 and A-Lab tasks are real-lab tasks with existing results, so the gap table is
non-empty at v0.
`scripts/ingest_results.py` adds completed lab episodes to the gate's training set with
`source="real"`; retraining runs only over those rows.

## 7. Evaluation

N = 200 test targets, stratified by tier, identical across agents, `seed = 0`.
Agents: Random (valid `add` set covering the target, one heat at uniform T, t, atm, stop);
Heuristic (training-fold modal precursor per element, T from a linear fit on mean precursor Tm,
12 h, air or reducing by the reducible-cation rule, regrind, reheat, xrd, stop);
Retrieval (nearest training recipe by Magpie cosine on the target, with subset and superset
chemsys excluded); Claude Opus 5.5 on Bedrock, plain and with retrieval, strict JSON, one retry
then heuristic fallback (counted), disk cache `cache/claude/<sha256>.json`, `MAX_CLAUDE_CALLS`
guard.

Metrics, in headline order:
1. Literature match, independent of the sim: final charged precursor set equals the held-out
   recipe's set (top-1), and max heat T within 150 C; T MAE against the median predictor
   (247 C, measured by the ML critique). Wilson 95% intervals; agents compared pairwise with McNemar.
2. Real-outcome AUC of the sim itself (§2 calibration), per source and macro-averaged, stratified
   by T. Agent-independent; reported with the heuristic as the grey curve.
3. Simulated success: `w_target >= 0.5` at stop, labelled "simulated". A reward-hack audit prints
   the mean number of heats and the share of episodes that stop without an XRD.
Outputs: `results/metrics.json`, `real_outcome_roc.png`, `lit_match.png`, `gap_by_task.csv`;
captions state the weak labels.

## 8. Module layout

```
synthgym/
  data.py      load_ceder(path) -> DataFrame; load_mp(path) -> dict[str, float]
               load_arrows3(path) -> list[RealRun]; load_alab(path) -> list[RealRun]
               split_chemsys(df, seed) -> tuple[DataFrame, DataFrame]
  thermo.py    class Phase(BaseModel): formula: str; e_form: float; tm_k: float | None
               build_library(target: str, mp: dict) -> list[Phase]
               enumerate_reactions(assemblage: dict[str, float], library) -> list[Reaction]
               class Reaction(BaseModel): reactants: dict[str, float]; products: dict[str, float]; de_per_atom: float
  gate.py      class Gate: fit(df_train) -> None; predict(rxn, T, t, atm, c, target) -> tuple[float, float]
               make_pseudo_negatives(df_train) -> DataFrame       # chemist point-4 set only
  sim.py       class Crucible: assemblage, contact, history; apply_heat(T, t, atm, gate) -> None
               weight_fractions(assemblage) -> dict[str, float]; xrd(assemblage, rng) -> dict[str, float]
  backends.py  class Backend(Protocol); class SimBackend; class LabQueueBackend
  env.py       class SynthesisGym(gym.Env): reset, step, to_json, parse_json, flags
  tasks.py     class TaskSpec(BaseModel): id, target, tier, split, library; list_tasks; make
  agents/      random.py, heuristic.py, retrieval.py, claude.py   # each: act(obs_json) -> dict
  evaluate.py  run(agent, tasks, seed) -> DataFrame; calibrate(gate) -> dict; wilson, mcnemar
scripts/       fetch_data.py, train_gate.py, calibrate.py, eval.py, demo.py, ingest_results.py
tasks/v0.json  results/  cache/claude/  lab/queue/  lab/results/
```

## 9. v0 cut (about 5 hours) and deferrals

| Hour | Deliverable | Check |
|---|---|---|
| 0.5 | `data.py` from the loader snippet; chemsys split | row counts printed |
| 1.5 | `thermo.py`: library, reaction enumeration, `dE` | YBCO charge yields `BaCuO2`, `Y2Cu2O5`, target |
| 2.5 | `gate.py`: labels, pseudo-negatives, LightGBM x8 | pseudo AUC with and without T |
| 3.5 | `sim.py`, `env.py`, spaces, JSON; `calibrate.py` | ARROWS3 600/700 C predict `w_target` near 0; AUC per stratum |
| 4.5 | Random, Heuristic, Claude plain; `eval.py` on 200 targets | `metrics.json`, both charts |
| 5.0 | `demo.py` from cached results | runs in under 60 s |

Deferred to v1: Retrieval agent and the with/without comparison; `LabQueueBackend`
polling (v0 writes the queue file and raises `Pending`); ordinal A-Lab AUC; `stochastic=True`;
the real-row retraining loop. Deferred to v2: oxygen chemical potential by atmosphere
(`mu_O(T, pO2)`) in `dE`; melt and solid-solution phases; particle-size dependence of `c`;
MACE-MP-0 energies for missing phases.

## 10. Risks and how they stay visible

1. **Missing formation energies.** 35.5% of Ceder recipes have full `e_form` coverage (chemist
   point 4). Uncovered targets are excluded; `calibrate.py` prints each task library's coverage.
2. **The gate is trained on one-step summaries of multi-step recipes.** `T_max` and summed time
   overstate what one step achieves. If the sim predicts full conversion at 800 C where ARROWS3
   shows `BaCO3` and `BaCuO2`, `gap_L1` at 800 C is large and is printed per stratum.
3. **Most-negative-`dE`-first ordering may be wrong under kinetic control.** It is the assumption
   with the least data behind it; the ARROWS3 assemblage L1 distance (not the AUC) measures it.
4. **Phase-library gaps.** Unknown or metastable products cannot appear, so the sim is
   optimistic. The 15 A-Lab Failed targets are the test: high predicted `w_target` for them
   lowers the target-feasibility AUC.
5. **Contact factor and the 0.5 decay are invented.** The regrind-reheat recipes (47% of Ceder)
   are the only evidence that it matters. The eval prints the `w_target` gain of a reheat with and
   without regrind, so a useless `c` shows.
6. **Reward hacks.** The §7 audit prints heats per episode, XRD-free stops and mean cost.

## Rationale

The phase-assemblage state was chosen because each component is observable in the real data
on disk, and the ARROWS3 pairwise scheme shows that enumeration plus a kinetic gate reproduces
intermediates. Rejected: history-in-observation with a
(target, action) oracle (a bandit); a learned latent state (uncalibratable); a kinetic model
with particle-size parameters (no data); terminating on success (leaks the hidden state and
removes the reason to characterise); Bernoulli rewards and two oracles (variance and a replica,
per the ML critique); melt, standalone carbonate and non-reducible atmosphere negatives (per the
chemist). The next step is hour 0.5 of §9.
