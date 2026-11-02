# Synthesis Gym

A reinforcement-learning environment for inorganic solid-state synthesis. An agent charges
precursors into a crucible, heats, regrinds, pays for XRD, and stops. The crucible is hidden: the
simulator tracks the true phase assemblage, and the agent sees only noisy XRD it pays for.
History changes the future: intermediates formed early consume driving force, and sintered powder
reacts slowly until it is reground.

The simulator is checked against real synthesis outcomes, including failures (ARROWS3, A-Lab), and
the backend interface is the same one a real furnace plugs into.

![demo](results/video/demo.gif)

## Run

```sh
uv sync
uv run python scripts/fit_kinetics.py          # fits the extent law on ARROWS3 (writes models/kinetics.json)
uv run python scripts/run_eval.py random heuristic claude
uv run python scripts/demo.py --task SynthGym-v0/hard/BaFe12O19 --agent claude
./scripts/record_demo.sh                        # results/video/demo.{cast,gif,mp4}
```

Claude runs on AWS Bedrock (`us.anthropic.claude-opus-5-5`, us-east-1, profile `default`). Replies
are cached in `cache/claude/`. Raw data goes in `data/raw/` (sources below).

## How the simulator works

- State: moles of each phase, a contact factor (halved by each heat, reset by regrind), mass lost
  as gas. See `synthgym/sim.py`.
- Reactions: for each phase and each pair of phases, a linear program finds the minimum-Gibbs-energy
  products over the Materials Project phase library (matminer `mp_nostruct_20181018`, 0 K formation
  energies). Gases leave with an entropy term; oxygen is exchanged with the atmosphere at a chemical
  potential set by its pO2 (air, O2, inert, vacuum, reducing). Carbonate decomposition, reaction with
  an acidic oxide below the standalone decomposition point, and reduction under H2 all come out of
  this one minimisation.
- Kinetics: extent `x = 1 - exp(-t * contact / tau)`, `tau = tau0 * exp(-beta (T/T_on - 1)) * exp(gamma dG)`,
  `T_on = alpha * min(Tm)`. Four scalars, fitted to ARROWS3 phase fractions (`synthgym/kinetics.py`,
  `scripts/fit_kinetics.py`).
- Reward: true target weight fraction at stop (+0.5 if >= 0.9) minus bounded process costs.

## Results (SynthGym-v0/test-200, held-out chemical systems)

| agent | precursor set = literature | set + T within 150 C | T MAE (C) | simulated w >= 0.9 | mean return |
|---|---|---|---|---|---|
| Claude Opus 5.5 (closed loop) | 64% [57, 70] | 46% [39, 53] | 139 | 83% [77, 88] | 1.17 |
| heuristic (modal precursor, Tamman T) | 58% [52, 65] | 33% [27, 40] | 202 | 62% [56, 69] | 0.95 |
| Qwen3-4B-Instruct base (plan mode) | 46% [39, 52] | 26% [20, 32] | 212 | 53% [46, 60] | 0.89 |
| random | 0% | 0% | 477 | 21% [16, 27] | 0.49 |

Brackets are Wilson 95% intervals, n = 200. Median-temperature predictor MAE: 244 C. Claude vs
heuristic, exact McNemar: recipe match p = 0.001, simulated success p = 2e-10, precursor match p = 0.09.
Literature columns do not read the simulator. "Literature" means published, purity unknown.

Simulator vs real outcomes (`results/kinetics_fit.json`, held out from the fit): YBCO precursor
sets, AUC of predicted target fraction vs observed >= 50 wt%: 0.79 pooled (n = 72); 0.62 within the
900 C stratum (n = 18). NTMO (never fit): 0.30. The simulator gets the temperature trend right and
the phase-level detail mostly wrong (mean L1 1.36 of a possible 2). These are the numbers to improve.

## Honest limits

- Formation energies are 0 K DFT from 2018 with no solid entropy. The task target is assumed to be
  the thermodynamic sink at its published temperature and atmosphere (the ARROWS3 premise); the
  shift is fixed per task, so other conditions can still destabilise it.
- Melting points are a hand table of round handbook values; unlisted phases use anion-class defaults.
- The contact factor (halved per heat) is an assumption with no calibration data.
- Test tasks need the target in the energy snapshot, which excludes doped and non-integer formulas.
- Open models are trained and evaluated in plan mode (whole recipe in one reply, open loop); Claude
  runs closed loop. Compare within a mode.

## Data sources

- Ceder group text-mined solid-state recipes, Kononova et al., Sci. Data 2019, release 2020-07-13
  (github.com/CederGroupHub/text-mined-synthesis_public). The repo has no licence file.
- ARROWS3, Szymanski et al., Nat. Commun. 2023 (github.com/njszym/ARROWS, Examples/*/Exp.json).
- A-Lab, Szymanski et al., Nature 2023, SI `41586_2023_6734_MOESM3_ESM.zip`.
- Materials Project formation energies via matminer `mp_nostruct_20181018`.

Design history: `docs/brief.md`, `docs/SPEC.md`, `docs/debate/`, `docs/arch/`, `docs/ARCHITECTURE.md`.
