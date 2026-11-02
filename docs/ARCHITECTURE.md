# Synthesis Gym v0: architecture

Status: adopted 2026-11-02 after a three-candidate Fable arena (docs/arch/candidate-{1,2,3}.md),
a Fable cross-judge, and round-1 critiques (docs/debate/). Supersedes docs/SPEC.md where they differ.
Synthesis note at the end.

## 1. The problem as an MDP

The crucible is the state. An episode is one sample: charge precursors, heat, regrind, add more,
measure, stop. History matters because each heat reacts whatever phases exist now: intermediates
formed at 800 C (BaCuO2, Y2Cu2O5 in YBCO) carry less driving force to the target than fresh
precursors, and a sintered powder reacts more slowly until it is reground.

Hidden state (`synthgym/sim.py: Crucible`):
- `moles: dict[str, float]`: moles of each phase per mole of target-equivalent charge.
- `contact: float in (0, 1]`: 1.0 after `add` or `regrind`; multiplied by 0.5 after each heat.
- `mass_loss: float`: fraction of charged mass released as gas.
- `history: list[Action]`.

Observation: target, step, cost so far, charged precursors, mass loss (free, a balance reading),
and the last paid XRD: weight fractions with N(0, 0.03) noise, phases under 3 wt% dropped, with its age.
`flags` are computed from the last XRD and the charge only, never from hidden state.

## 2. Transition (`synthgym/sim.py`, `synthgym/kinetics.py`)

`heat(T_c, t_h, atm)`, in order:

1. Decompose (assumption, table): carbonates, nitrates, hydroxides, ammonium salts release gas above
   an onset. Carbonate onsets are lowered when an acidic oxide (Si, Ti, P, Mo, W, V, Nb, Te) is present
   (BaCO3 700, SrCO3 750, CaCO3 650, Li2CO3 450, Na2CO3 500 C), else standalone values.
2. React (physics + fitted extent), three passes. For every pair of solid phases present, find the
   product set from the task phase library that balances the pair (scipy `nnls`, with O2 as a free
   reactant or product) and minimises
   `dG = dE_form - n_gas * 2.1e-3 * T_K / n_atoms` (eV/atom). Energies: matminer `mp_nostruct_20181018`
   lowest polymorph per reduced formula, plus a hand table for `Y2(CO3)3` and `BaYCu3O7`. Skip if
   `dG >= -0.02`. Apply pairs most negative `dG` first, each with extent
   `x = 1 - exp(-t_h * contact / tau)`, `tau = tau0 * exp(-beta * (T_K / T_on - 1)) * exp(gamma * dG)`,
   `T_on = alpha_class * min(Tm_K of the pair)`.
3. Atmosphere gate (assumption): products with reducible cations (Cu, Ni, Co, Fe3+, Pb, Bi, Ag,
   Mn4+, V5+, Mo6+, W6+) are blocked under `reducing`.
4. `contact *= 0.5`.

Learned, and only these: `tau0, beta, gamma` fitted by `scipy.optimize.minimize` on ARROWS3 phase
weight fractions (fit: LTOPO + half the YBCO precursor sets; held out: the other YBCO sets + NTMO);
`alpha_class` per anion class as the 10th percentile of first-heat `T_K / min Tm_K` on the Ceder
training split. No pseudo-negatives anywhere in v0. Fit result: `results/kinetics_fit.json`.

Phase library per task: `mp_nostruct` formulas inside the target chemsys plus C, H, N with
`e_hull <= 0.10`, the charged precursors, and gases CO2, H2O, NH3, NO2, O2. Tasks whose target or
literature precursors lack an energy are excluded from v0 suites and counted.

## 3. Actions and spaces (`synthgym/env.py`)

LLM / JSON form, parsed by `parse_action(dict) -> Action | Invalid`:

```json
{"op": "add", "precursors": {"BaCO3": 2, "Y2O3": 0.5, "CuO": 3}}
{"op": "add", "precursors": ["BaCO3", "Y2O3", "CuO"]}
{"op": "heat", "temperature_c": 900, "time_h": 12, "atmosphere": "air"}
{"op": "regrind"}  {"op": "xrd"}  {"op": "stop"}
```

A list for `add` means "balance it for me" (nnls to target stoichiometry). `add` is legal at any step
and resets contact. Gymnasium `Dict` spaces mirror this (op Discrete(5), precursors MultiBinary(V),
T Box[200,1700], log time Box[log 0.5, log 72], atmosphere Discrete(5): air, O2, inert, reducing, vacuum).

Invalid (cost 0.05, state unchanged): precursor outside the training vocab; a target element
(other than O, H, C, N) not supplied; a precursor bringing a non-volatile element absent from the
target; any single precursor containing every non-volatile element of the target (blocks
"charge the target"); heat before add.

## 4. Reward and termination

- Heat: `-(0.02 + 0.08 * (T_c / 1700) * log1p(t_h) / log1p(72))`; 900 C x 12 h = -0.054.
- Regrind -0.01, add -0.01, xrd -0.02. Process cost capped at -0.40 per episode.
- Terminal, at `stop` or truncation (max 10 steps): `w_target + 0.5 * [w_target >= 0.9]`, the true
  weight fraction. Deterministic by default.
- Never terminates on success: stopping is the agent's call, which is what gives XRD its value.

## 5. Tasks and registry (`synthgym/tasks.py`)

Split by chemsys 85/15, seed 0; train chemsys that are subsets or supersets of a test chemsys leave
train; Ba-Cu-O-Y, Li-O-P-Ti, Mo-Na-O-Te and every A-Lab chemsys leave train. Vocab, frequency tables,
Tamman fit and retrieval index come from train only.

Tiers from data: `easy` (two cations, oxide precursors, one literature heat), `medium` (three+
cations or a carbonate/nitrate/phosphate precursor or two heats), `hard` (reducible cation, volatile
cation, or a library intermediate within 0.03 eV/atom of the target).

Ids: `SynthGym-v0/easy/BaTiO3`; suites `SynthGym-v0/test-200`, `SynthGym-v0/real/arrows3`,
`SynthGym-v0/real/alab`. `tasks/v0.json` pins split seed, vocab hash, constants and target lists.
API: `make(task_id, backend=None) -> SynthesisGym`, `list_tasks(suite) -> list[str]`.

## 6. Backends and the real-lab tier (`synthgym/backends.py`)

```python
class Backend(Protocol):
    def reset(self, task: TaskSpec) -> None: ...
    def apply(self, action: Action) -> StepResult: ...   # StepResult(xrd: dict | None, pending: bool)
```

- `SimBackend`: section 2.
- `ReplayBackend`: ARROWS3 as a furnace stand-in. A heat is legal only at a recorded (precursor
  set, T); `xrd` returns the recorded products. The same episode is mirrored through `SimBackend`
  and each step's pair is written to `results/gap/<task>.jsonl`.
- `QueueBackend`: writes `lab/queue/<episode>/<step>.json`, returns `pending`; results come back as
  `lab/results/<episode>/<step>.json` with `{"phases": {...}}`. Ingested rows carry `source="real"`;
  the only retraining loop allowed runs on those rows.

Sim-to-real gap per task: `gap_L1` = mean over XRD steps of the sum of |w_sim - w_real|, plus
`gap_target` = |w_target_sim - w_target_real| at the end.

## 7. Evaluation (`synthgym/evaluate.py`)

Headline, independent of the simulator the agent optimises:
1. Real-outcome AUC of the sim: predicted `w_target` after each ARROWS3 recorded protocol vs observed
   `>= 50 wt%`, per source and per temperature stratum (YBCO 900 C: 28 vs 10), held-out sets only.
   A-Lab target-feasibility AUC (Success 35 vs Failed 15; Partial in an ordinal AUC).
2. Literature match on test-200: first `add` set equals the literature set, hottest heat within
   150 C; temperature MAE against the median predictor. Wilson 95% intervals, McNemar between agents.

Secondary, labelled "simulated": terminal reward, `w_target >= 0.9` rate per tier; reward-hack audit
(heats per episode, stops without XRD, mean cost). Claude reported with and without flags and with
and without retrieval.

Agents (`synthgym/agents.py`, each `act(obs_json) -> dict`): Random; Heuristic (modal train precursor
per element, Tamman fit for T, calcine T-300 C, regrind, sinter, xrd, stop); Retrieval (nearest train
recipe by Magpie cosine, subset/superset chemsys excluded); Claude Opus 5.5 on Bedrock (us-east-1,
`us.anthropic.claude-opus-5-5`, personal account), strict JSON, retry once then Heuristic (counted),
cache `cache/claude/<sha256>.json`, `MAX_CLAUDE_CALLS`.

## 8. Module layout

```
synthgym/
  data.py       Ceder loader (done)            thermo.py   energies, balance, dG (done)
  realdata.py   ARROWS3 + A-Lab (done)         kinetics.py Kinetics params, extent, fit
  sim.py        Crucible, heat, xrd            backends.py Sim, Replay, Queue
  env.py        SynthesisGym, parse_action     tasks.py    split, tiers, registry
  agents.py     Random, Heuristic, Retrieval, Claude
  evaluate.py   run, real_outcome_auc, wilson, mcnemar, plots
scripts/ fit_kinetics.py  run_eval.py  demo.py
```

## 9. v0 order (each ends in a check)

1. `kinetics.py` + `sim.py` with prior constants. Check: YBCO charge gives ~0 target at 600/700 C and
   intermediates first.
2. `fit_kinetics.py`. Check: held-out L1 and stratified AUC in `results/kinetics_fit.json`.
3. `tasks.py`, `env.py`, `backends.py` (Sim, Replay; Queue writes files). Check: `check_env` passes.
4. Agents. Check: 200 Heuristic episodes in under a minute.
5. `run_eval.py` with Claude on test-200, charts, `demo.py` replaying cached results.

Deferred: learned extent model on `source="real"` rows; mu_O(T, pO2) in dG; melts and solid
solutions; MACE energies for missing phases; heating rate and particle size in state.

## 10. Risks, each with the artifact that shows it

- Kinetics fit on two chemsys: held-out L1 per source in `kinetics_fit.json`; NTMO is never fit.
- Most-negative-dG ordering misses kinetic paths: `results/missed_phases.json` lists phases ARROWS3
  observed that the sim never produced.
- Energy coverage: tasks excluded for missing energies are counted in `metrics.json`.
- `contact` is uncalibrated: an ablation with contact frozen at 1 is in `metrics.json`.
- Positives mean "published", not phase-pure: every caption says so.

## Synthesis note

- Base: candidate 1 (physics transition, three-scalar extent law fitted on ARROWS3, no
  pseudo-negatives). The cross-judge chose the same base independently (24 / 24 / 23; tie with
  candidate 2 broken on buildability and held-out calibration).
- Grafts from candidate 2: `ReplayBackend` and per-step gap log; acidic-oxide carbonate onsets;
  three passes over pairs; moles as the internal unit; Claude reported with flags off.
- Grafts from candidate 3: `add` at any step; flags from observed XRD only; never terminate on
  success; reward-hack audit; driving-force-trap tier rule.
- From the cross-judge: block any precursor that alone contains every non-volatile target element;
  hand energies for `Y2(CO3)3` and `BaYCu3O7` (both measured missing).
- Rejected: candidate 2's convex hull and InterfacialReactivity core and nine-parameter fit over
  Ceder (each loss evaluation re-simulates about 15k recipes, infeasible tonight); candidate 3's
  LightGBM gate (labels are one-step summaries of multi-step recipes); success termination (leaks
  hidden state).
- Convergence: all three candidates chose the phase-assemblage state, paid XRD, regrind via a contact
  factor, true target fraction as reward and a shared backend protocol. Strong agreement; no reframe.
