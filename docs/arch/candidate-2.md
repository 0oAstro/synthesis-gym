# Synthesis Gym architecture, candidate 2: phase-assemblage MDP

## 0. Thesis

(measured) means computed from `data/raw/`. The state is the powder in the crucible: a phase assemblage plus a hidden reactivity scalar. A
heating step reacts the assemblage pairwise along driving forces from Materials Project formation
energies, gated by a kinetic onset rule fitted on ARROWS3 intermediates and Ceder recipes.
Intermediates consume driving force, volatiles leave, grains coarsen until regrind, and the agent
sees the assemblage only when it pays for an XRD. Each makes the transition path dependent.
19,070 of 31,782 raw Ceder recipes have two or more heats, 9,385 end hotter than they start,
4,520 contain heat, grind, heat (measured).

## 1. State model

### 1.1 Hidden state

```
phases:     dict[str, float]  # formula -> moles per formula unit of target
reactivity: float in [0, 1]   # fresh powder 1; decays with heating; regrind resets
lost:       dict[str, float]  # moles of volatile component lost
spent:      float             # cumulative cost charged
```

The task phase list is the target, train-vocab precursors inside the chemsys plus C, H, N,
`mp_nostruct` formulas in the chemsys with `e_hull <= 0.03 eV/atom`, and the gases CO2, H2O, NH3,
NO2, O2, padded to `K = 32`.

### 1.2 Observation

The agent sees the target, its history, the cost spent, and the last XRD it paid for. The XRD
reports weight fractions above 3 wt% and goes stale when the next heat runs (`xrd_age`).
`reactivity` and `lost` are never observed; a stalled reheat or an off-stoichiometry XRD is their
only evidence.

### 1.3 Alternatives

**A, history in the observation, oracle on (target, action).** SPEC v1: the transition ignores
history, so the optimum is one argmax action repeated. Rejected by owner decision.

**B, a scalar reaction coordinate** `x in [0,1]` advanced by a learned `f(x, T, t)`. Sequential,
but it cannot represent ARROWS3: in LTOPO the target fraction is non-monotone in temperature in 13
of 15 precursor sets because Li3PO4, Li4P2O7 and TiP2O7 form first and compete (measured). A
scalar cannot name the wrong phase, so "add a precursor" means nothing.

**Chosen, phase assemblage.** It is what ARROWS3 records (products and weight fractions per
precursor set per temperature: YBCO 47 sets x 5 T, 200 rows, 149 verified; LTOPO 58 rows; NTMO
18), so the sim is scored on it row by row; its intermediates are real (BaCuO2 in 80 YBCO rows,
BaY2CuO5 in 49, Y2Cu2O5 in 35, measured). Regrind, add-precursor, atmosphere and XRD become decisions.

## 2. Transition model

A heat step runs Algorithm 1; lines are tagged **P** physics, **L** learned, **A** assumption.

```
heat(S, T, t, atm):
 1. decompose (A, table): carbonate -> oxide + CO2 above the reaction onset when an acidic
    oxide (P, Si, Ti, Mo, W, V, Nb) is present (BaCO3 700, SrCO3 750, CaCO3 650, Li2CO3 450,
    Na2CO3 500 C), else the standalone value; hydroxide, nitrate, ammonium phosphate at
    300/400/250 C.
 2. three passes over pairs (A, B) of present phases, most negative dE first:
        products, dE = reaction(A, B, hull, atm)       # P, 2.1
        if dE >= 0: continue
        xi = gate(dE, T, t, A, B) * S.reactivity         # L, 2.2
        move xi * limiting moles from (A, B) to products
 3. volatilise (A): for c in {Li2O, Na2O, K2O, PbO, Bi2O3, MoO3, TeO2}:
        lost += moles(c) * (1 - exp(-t / 24)) * sigmoid((T - T_vol(c)) / 50)
 4. coarsen (A): reactivity *= exp(-(t / 12) * sigmoid((T - 0.6 * Tm_min) / 100))
```

### 2.1 Physics

`hull` is a pymatgen `PhaseDiagram` over `PDEntry(Composition(f), e_form * n_atoms)` for the
`mp_nostruct` entries in the chemsys plus gas references. Products of (A, B) are the hull
decomposition at the mixing ratio maximising `-dE` (`InterfacialReactivity`), eV/atom at 0 K. Entropy enters only through gases: releasing CO2, H2O or NH3 earns `-T * 1.9e-3 eV/K` per
molecule (**A**). Atmosphere is a per-cation redox shift (**A**): under `reducing` the e_form of
oxides of Cu2+, Ni2+, Co, Fe3+, Pb2+, Bi3+, Ag+, Mn4+, V5+, Mo6+, W6+ rises 0.3 eV/atom so the lower
oxide or metal wins; `O2` shifts the same list -0.05; `Ar`, `N2`, `vacuum` remove the O2 sink.
`mp_nostruct` covers 35.5% of recipes fully (SPEC); a phase missing from the hull is a spectator
with `dE = 0` and the env records `info["missing_energy"]`.

### 2.2 Learned gate

```
gate(dE, T, t, A, B) = sigmoid(k * (T - T_on)) * (1 - exp(-t / tau))
T_on = alpha * min(Tm_A, Tm_B) + beta * dE + gamma[anion_class(A, B)]
```

Nine parameters: `alpha` (Tamman, prior 0.67), `beta` (prior -300 C per eV/atom), `gamma` for
{oxide-oxide, carbonate, phosphate, other}, `k` (prior 1/80 per C), `tau` (prior 4 h). `Tm` is a
100-row compound melting-point table (87.7% of precursor occurrences); missing phases take their
class mean. Melting is never a failure flag: 49.7% of real recipes exceed a precursor's Tm.

**Labels.** (i) ARROWS3: product set and weight fractions per (precursor set, T), 276 rows, three
chemsys. (ii) Ceder train recipes with per-step T and time: the target reaches `w >= 0.5` after the
final heat (positive); the same recipe with the final T lowered 250 C or the final time cut to
1/20 does not (balance-preserving pseudo-negatives, chemist point 4; melt, standalone carbonate
and stable-oxide atmosphere negatives are out). Loss: cross-entropy on `w_target >= 0.5` plus
`0.2 *` L1 weight-fraction error on ARROWS3 rows, Nelder-Mead over nine parameters. v1 replaces
`T_on` with a LightGBM regressor on (dE, Tamman ratio, anion classes, cation set, log t, atm) on
the same labels; v0 stays parametric: 276 real rows support no more.

### 2.3 Calibration, `results/sim_calibration.json`

Per ARROWS3 chemsys: phase-set Jaccard per row; target-fraction MAE; AUC of simulated `w_target`
against observed `>= 50%` per temperature stratum (YBCO verified: 600 C 0 of 25, 700 C 0 of 36,
800 C 3 of 39, 900 C 28 of 38, 1000 C 2 of 11, measured). The 900 C stratum (28 vs 10) is the
recipe-level number; pooled, it is temperature in disguise. A-Lab: 57 rows, 33 Success,
2 Success (offline), 7 Partial, 15 Failed (measured); every recipe is 4 h and Failed rows have
none, so the metric is target-level: simulate the Heuristic recipe at 700, 900 and 1100 C for 4 h
in air, take `max w_target`; AUC Success vs Failed (35 vs 15) plus an ordinal AUC with Partial in
the middle.

## 3. Spaces

```python
V = len(train_vocab); K = 32; H = 8          # train-fold precursors, phase slots, max steps
action_space = Dict({
    "op":          Discrete(5),              # charge, heat, regrind, xrd, stop
    "precursors":  MultiBinary(V),           # charge only; env balances stoichiometry by NNLS
    "temperature": Box(200.0, 1700.0, ()),
    "log_time":    Box(log(0.5), log(72.0), ()),
    "atmosphere":  Discrete(6),              # air O2 Ar N2 reducing vacuum
})
observation_space = Dict({
    "target_elements": Box(0, 1, (118,)),  "target_magpie": Box(-inf, inf, (132,)),
    "phase_names":     MultiDiscrete([V + K] * K),
    "xrd_wt":          Box(0, 1, (K,)),  "xrd_age": Discrete(H + 1),   # H means never
    "history":         Box(-inf, inf, (H, 8)),   # op, T, log t, atm, n_prec, cost, step, valid
    "spent":           Box(0, 1, ()),        "step": Discrete(H + 1),
})
```

`charge` at step 0 sets the precursors; later it adds one sized to the deficit. Invalid
actions (precursor outside vocab, target element unsupplied, non-volatile foreign element, heat
before charge) cost 0.1 and skip the furnace. The LLM form uses names:

```json
{"op": "charge", "precursors": ["BaCO3", "Y2O3", "CuO"]}
{"op": "heat", "temperature_c": 900, "time_h": 12, "atmosphere": "air"}
{"op": "xrd"}  {"op": "regrind"}  {"op": "stop"}
```

The observation JSON carries `target`, `history`, `last_xrd: [{"phase": "BaCuO2", "wt": 0.31}, ...]`,
`xrd_age`, `spent` and `flags` such as `"BaCO3 persists: 700 C is below its onset with CuO"`,
fed back verbatim to Claude (YC point 3).

## 4. Reward and termination

```
r_t = -cost(a_t)                                        every step
cost(heat)   = min(0.06, 0.012 * (T / 1000) ** 2 * log1p(t_h))   # 1100 C x 12 h = 0.037
cost(xrd)    = 0.02      cost(regrind) = 0.01      cost(charge) = 0.0
cost(invalid)= 0.10
r_T = w_target(S_T) + 0.5 * [w_target >= 0.9]           at stop or truncation
```

`w_target` is the true weight fraction in the hidden state, not a Bernoulli draw. Total cost over
8 steps is at most 0.5 against a +1.5 ceiling: stopping at step 0 yields 0, a 30-minute anneal
yields what the gate allows (`1 - exp(-0.5/4) = 0.12`), and a second 1400 C x 48 h sinter costs
0.06, paying only if it raises `w_target` by more. Termination: `stop` or `w_target >= 0.99`;
truncation at `H = 8`. `stochastic=True` adds Gaussian noise (sd 0.03) to XRD weights and the
gate; default off. `info` carries true `w_target`, the assemblage, `reactivity`, `flags`.

## 5. Task suite and registry

A task is `(target, phase list, literature recipe, tier)`. Targets are the 10,582 clean Ceder
targets after the chemsys split: 85/15 by sorted element set, seed 0; train chemsys that are a
subset or superset of a test chemsys, and ARROWS3 and A-Lab chemsys, leave train. Test is sampled
to 200 targets stratified by tier. Tiers:

- `easy`: 2 cations, no carbonate, no volatile or redox-flag cation, one literature heat.
- `medium`: 3 or more cations, or a carbonate precursor, or two heats.
- `hard`: a volatile component, a redox-flag cation, or a competing hull phase within 0.02 eV/atom
  of the target.
- `real`: the three ARROWS3 systems, replayable (Section 6).

Ids: `SynthGym-v0/easy/BaTiO3`, `SynthGym-v0/hard/LiFePO4`, `SynthGym-v0/real/YBCO`. The version
pins split seed, vocab hash, gate parameters and phase-list rule. Registry:
`list_tasks(tier=None, split="test") -> list[str]`, `make(task_id, backend="sim", **kw) ->
SynthesisGym`, `gymnasium.register("SynthGym-v0", ...)`.

## 6. Real-lab tier

```python
class Backend(Protocol):
    def charge(self, precursors: dict[str, float]) -> str            # sample_id
    def heat(self, sample_id, T, t_h, atm) -> Ticket
    def regrind(self, sample_id) -> Ticket
    def xrd(self, sample_id) -> Ticket
    def poll(self, ticket) -> XRDResult | None                       # None while pending
SimBackend     # Algorithm 1, immediate
ReplayBackend  # ARROWS3: heat legal only at a recorded (set, T); xrd returns the recorded products
QueueBackend   # writes lab/queue/<ticket>.json, reads lab/results/<ticket>.json
```

`step` with a pending ticket returns `obs` unchanged with `info["pending"] = True`; the furnace
result arrives as the XRD observation, as in sim. Every backend call also runs through `SimBackend`
on the same hidden state; the pair is logged to `results/gap/<task_id>.jsonl` as
`{step, sim_wt, real_wt, jaccard, abs_err_target}` and the per-task gap is their mean.
`ReplayBackend` computes it tonight on 276 real rows through the furnace's code path. Rows from
`lab/results/` join the gate's labels with `source="real"`; retraining runs only on those rows
(owner decision), charted as AUC against rows ingested.

## 7. Evaluation protocol

N = 200 held-out targets, all agents on the same targets, seed 0, `H = 8`. Primary, independent of
the sim:
1. Literature match: charge set equals the literature precursor set (reduced formulas) and the
   hottest heat is within 150 C of the literature max T; first charge and final episode. Wilson
   95% intervals, McNemar between agents; 36% vs 60% at alpha 0.05, power 0.8 needs 67 pairs.
2. Temperature MAE against literature; the median predictor (247 C) is the floor.
3. Real-outcome AUC of the sim (2.3): ARROWS3 per stratum and chemsys, A-Lab target-level, n in
   every caption.

Secondary, labelled "simulated": final `w_target`, episodes at 0.9, XRD and regrind counts.

Baselines, re-run on this split: Random (covering precursors, uniform T, t, atm, random ops);
Heuristic (modal train precursor per element, Tamman linear fit for T, 12 h, air unless a
redox-flag cation, one XRD, one regrind and reheat 100 C hotter if `w < 0.9`); Retrieval (nearest
train target by Magpie cosine excluding subset or superset chemsys, copy its recipe); Claude Opus
5.5 plain and with retrieval, strict JSON, one retry then Heuristic fallback (counted). Claude
calls are cached at `cache/claude/<sha256>.json`, capped at `MAX_CLAUDE_CALLS = 2000` (under 40
USD). All agents are reported with and without retrieval.

## 8. Module layout

```
synthgym/
  data.py      load_ceder(path) -> DataFrame              # per-step heats as list[HeatStep]
               load_mp(path) -> dict[str, (e_form, e_hull)];  load_arrows3(dir) -> list[ArrowsRow]
               load_alab(path) -> DataFrame;  split_chemsys(df, seed) -> Split
  thermo.py    build_hull(chemsys, mp) -> PhaseDiagram
               reaction(a, b, hull, atm) -> (products: dict[str, float], dE: float)
               decompose(phase, T, present) -> dict[str, float] | None
  kinetics.py  GateParams(alpha, beta, gamma, k, tau); gate(dE, T, t_h, a, b, p) -> float
               fit_gate(rows, recipes, seed) -> GateParams
  state.py     Assemblage(phases, reactivity, lost, spent)          # frozen; heat returns a new one
               heat(s, T, t_h, atm, p, hull) -> (Assemblage, flags: list[str])
               xrd(s, noise) -> dict[str, float];  weight_fraction(s, target) -> float
  backend.py   Backend, SimBackend, ReplayBackend, QueueBackend, Ticket, XRDResult
  env.py       SynthesisGym(task, backend, stochastic=False, seed=0): reset, step, to_json, from_json
  registry.py  Task, list_tasks, make, tier_of(target, hull, recipe) -> str
  agents.py    RandomAgent, HeuristicAgent, RetrievalAgent, ClaudeAgent    # act(obs_json) -> action
  eval.py      run(agents, task_ids, seed) -> DataFrame; wilson(k, n); mcnemar(a, b)
  calibrate.py arrows_report(p) -> dict; alab_report(p) -> dict
scripts/ fetch_data.py train_gate.py eval.py demo.py;  results/ sim_calibration.json metrics.json gap/
```

## 9. v0 in 5 hours

1. `data.py`: per-step heats, `split_chemsys`, `load_mp`, `load_arrows3` with reduced-formula
   normalisation (`Y2(CO3)3`, `NaHO`, `PH9(NO2)2`). Done: 20,707 clean rows, 276 ARROWS3 rows.
2. `thermo.py`, `state.py` with prior `GateParams`. Done: BaCO3 + Y2O3 + CuO gives no target at
   600 C, `w > 0.5` at 900 C.
3. `fit_gate`, `calibrate.py`, `sim_calibration.json`, `real_outcome_roc.png`. Done: AUC per
   stratum with n.
4. `env.py`, `registry.py`, `SimBackend`, `ReplayBackend`, three baseline agents. Done:
   `env_checker` passes; 200 Heuristic episodes under a minute.
5. `ClaudeAgent` with cache and cap, `eval.py` on 200 targets, `metrics.json`, `demo.py`. Done:
   demo prints steps, flags, final XRD, two charts.

Deferred to v1: LightGBM gate; `QueueBackend` beyond a docstring and two directories; XRD noise
tuned to real patterns; precursor Magpie; a fuller redox table. v2: particle size and heating rate
in the state; MACE energies for phases missing from `mp_nostruct`; flux routes; retraining on
`source="real"` rows once a furnace runs.

## 10. Risks

1. **Incomplete hull** (35.5% full coverage): spectators with `dE = 0` make the sim optimistic.
   Visible: `info["missing_energy"]`; match rates split by coverage.
2. **Gate fitted on three chemsys**: transfer to fluorides or nitrides is unmeasured. Visible:
   tiers `real` and `hard` reported apart; A-Lab (n 50) is the only cross-chemsys check.
3. **Reactivity decay and volatilisation are unlabelled assumptions.** Visible: named constants in
   `state.py`; `calibrate.py` prints whether the sim needs a regrind where the 4,520
   heat-grind-heat recipes used one.
4. **Pseudo-negatives shape the gate.** Visible: the ARROWS3-only fit is printed beside the full fit.
5. **A 0 K hull misorders close phases.** LTOPO's Li4P2O7 versus Li3PO4 is the test; Jaccard per
   LTOPO row is printed, not averaged.
6. **Claude can overfit to the flags.** Visible: literature match does not read the sim; Claude is
   also reported with flags off.

## Rationale

Rejected: the contextual bandit (owner decision); a scalar progress state (cannot name the wrong
phase; 13 of 15 LTOPO sets are non-monotone); a next-state regressor on ARROWS3 alone (276 rows
for a 32-wide output); an LLM as simulator (unfalsifiable; the headline must not depend on the
model the agent optimises); MACE or DFT kinetics (no GPU, still no kinetics). The phase-assemblage
MDP with a thermodynamic hull and a nine-parameter gate is the smallest model that is sequential,
chemically legible, scored row by row against real intermediates, and furnace-swappable without
changing the agent.

Current state: document only. Next step: hour 1 of Section 9.
