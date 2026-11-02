# Synthesis Gym: build spec v1 (draft for debate)

Status: DRAFT, pre-debate. Inputs: `docs/brief.md` plus three research passes run on 2026-11-02
(data, libraries, prior art). Facts marked (verified) were observed in a run on aardvark.

## 0. One-line thesis

A Gymnasium environment for solid-state synthesis where an agent proposes recipes and a
calibrated learned oracle scores them; the oracle is already checked against real lab outcomes,
including real failures, and the architecture has a slot for a real furnace.

## 1. Stack (verified install on aarch64, Python 3.13, uv)

anthropic[bedrock] 2.3, gymnasium 1.4, lightgbm 4.7, matminer 0.10.1, matplotlib 3.11,
numpy 2.5, pandas 2.3, pydantic 2.14, pymatgen 2026.10.24, pytest 9.2, rich 15, scikit-learn 1.9.
Not used: mp-api (no MP_API_KEY), mace-torch (5.6 GB, stretch only).

Claude: `AnthropicBedrock(aws_region="us-east-1")`, model `us.anthropic.claude-opus-5-5`,
credentials from the AWS instance role (profile `default`).

## 2. Data

| Source | Use | Size (verified) |
|---|---|---|
| Ceder solid-state 2020-07-13 (`solid-state_dataset_20200713.json.xz`, repo root, no licence file) | positives, precursor vocab, retrieval | 31,782 raw; 20,707 clean; 15,145 with T and time; 10,582 targets; 1,689 precursors |
| matminer `mp_nostruct_20181018` | offline formation energy per formula (eV/atom) for reaction energy | 83,989 entries, 3 MB |
| ARROWS3 `Examples/{YBCO,NTMO}/Exp.json` | real outcomes incl. failures | YBCO 149 verified runs (72 no target, 44 trace, 33 >=50%); NTMO 18 |
| A-Lab SI `alab_synthesis_results.csv` | real outcomes incl. failures | 56 targets: 35 success, 7 partial, 15 failed |

Loader: port `/var/tmp/synthgym-research/data/loader_snippet.py` (tested). Use the per-material
`composition[].elements` dicts the dataset already carries instead of re-parsing strings with
pymatgen; drop rows whose compositions do not resolve to numbers. Atmosphere enum:
`air, O2, Ar, N2, reducing, vacuum, unspecified` (rare CO, CO2, NH3, He fold into nearest class or drop).
Temperature is the max heating T in C; time is summed heating hours.

Data lives in `data/raw/` (gitignored), fetched by `scripts/fetch_data.py` from the URLs in §2.

## 3. Split (leak control)

No publication year in the records (DOI only), so the year split of He et al. 2023 is out tonight.
Split by **chemical system** (sorted element set of the target), not by formula: doped and
rounded variants of one compound share a chemsys, so they cannot straddle the split.
85/15 by chemsys, seeded. Remove every chemsys that appears in ARROWS3 or A-Lab targets from
training, so the real-outcome set is untouched.

## 4. Oracle v0

Features per (target, recipe):
- composition: Magpie ElementProperty (matminer) of target; mean of precursors.
- process: T_c, log10 time_h, atmosphere one-hot, n_precursors.
- physics:
  - Tamman ratio `T_K / mean(Tm_K precursors)` (Huo 2022) and onset flag `T_K < 2/3 min Tm_K`.
  - melt flag: T above Tm of any precursor (elemental proxy for compound Tm where missing).
  - carbonate flag: carbonate precursor with T below its decomposition T (CaCO3 825, SrCO3 1000, BaCO3 1050, Li2CO3 720, Na2CO3 850 C).
  - reaction energy: dE per atom from mp_nostruct, reactants = precursors, products = target + volatiles (CO2, H2O, NH3, NO2, O2), stoichiometry from a least-squares balance; NaN when any formula is missing. 0 K, no entropy: feature only.
  - element-balance flag (hard rule, see §5).

Labels: real recipe = 1. Pseudo-negatives = 0, generated per positive:
1. swap one precursor for another vocab compound with the same cation (e.g. Li2CO3 -> LiNO3 is NOT a negative; swap to a compound of a different cation that keeps the element set broken) ;
2. T shifted by -400 to -600 C or above the precursor melt;
3. drop a precursor that supplies a target element (fails balance);
4. reducing-required target in air, or air-required oxide in H2.
This is PU-style weak labelling. README and every chart caption say so.

Model: LightGBM, 8 bootstrap members, output mean p and std. Isotonic calibration on a
validation chemsys fold.

Two oracles, trained on disjoint halves of the training chemsys:
- `env_oracle` drives rewards the agent sees.
- `judge_oracle` scores "sim success" for evaluation; the agent never queries it.
This blunts the circularity of scoring an agent with the model it optimises.

Oracle report (`results/oracle_report.json`): ROC-AUC on held-out pseudo-labels, calibration
plot, and **ROC-AUC on real outcomes** (ARROWS3 + A-Lab, success vs failure). The real-outcome
AUC is the number that matters; the pseudo-label AUC is a sanity check.

## 5. Environment

`SynthesisGym(target, max_steps=5, seed, oracle, real=False)` with Gymnasium API.

Action (pydantic-validated dict; flat wrapper for SB3-style libs):
`{precursors: list[str] (from vocab, 1-5), temperature_c: float [200,1700], time_h: float [0.5,72],
atmosphere: enum}` or `{stop: true}`.

Hard validity (invalid -> reward -0.5, no oracle call):
- every target element except O, H, C, N is supplied by some precursor;
- no precursor brings a non-volatile element absent from the target;
- precursors in the vocab.

Step: oracle -> p, std; success ~ Bernoulli(p) with env RNG. Reward = +1 success - 0.05 per
step - 0.0002 * T_c * time_h / 100 (energy proxy, rescaled so a 1000 C x 12 h step costs 0.024 C... see open question Q4).
Terminates on success or stop; truncates at max_steps. info carries p, std, validity reasons.
Observation: target formula, element vector, history of (action, p, std, success), best_p.
`real=True`: `lab.submit(action)` writes `lab/queue/*.json`, returns `pending`; `lab.ingest()`
reads `lab/results/*.json` into training data with `source="real"`.

## 6. Agents

- Random: random vocab precursors covering the target elements, uniform T/time/atm.
- Heuristic: most frequent training precursor per target element (He 2023 baseline, published
  top-1 36%), T from a linear fit of T on mean precursor Tm on training data (Huo 2022), 12 h, air
  unless oxidation-state rule says inert/reducing.
- Claude (Opus 5.5 on Bedrock): prompt = target, per-element allowed precursors (top 15 by
  frequency), history with oracle p/std, strict JSON. Retry once, then heuristic fallback (logged,
  counted in metrics). Modes: `plain` and `retrieval` (5 nearest training recipes by Magpie
  cosine on target).

## 7. Evaluation

Held-out set: 50 targets sampled from test chemsys (cost cap).

Primary (independent of the oracle): on held-out literature recipes,
- top-1 exact precursor-set match on the agent's first proposal and on its final proposal,
- temperature MAE (C) vs literature,
- compare against He 2023 most-frequent (36% top-1) and Huo 2022 (~140 C MAE).

Secondary: oracle real-outcome ROC-AUC (ARROWS3 + A-Lab).

Tertiary, labelled "simulated": success rate within max_steps scored by `judge_oracle`, per
agent, and per round of the sim-only loop (round k: run agents, add their episodes to env_oracle
training with `source="sim"`, retrain).

Outputs: `results/success_by_agent.png`, `results/success_by_round.png`,
`results/oracle_calibration.png`, `results/metrics.json`. Large fonts, captions state weak labels.

## 8. Repo layout

As in brief §9 plus `scripts/fetch_data.py`, `data/raw/`, `lab/queue/`, `lab/results/`.

## 9. Open questions for the debate

Q1. Is the env_oracle / judge_oracle split enough to make "sim success" meaningful, or should the
    chart lead only with literature match + real-outcome AUC?
Q2. The sim-only "rounds" loop: does retraining the oracle on its own Bernoulli samples teach
    anything, or is it a self-confirming loop that should be cut?
Q3. Which pseudo-negative types are actually informative vs trivially separable (the model learns
    "T < 300 C" and nothing else)?
Q4. Reward scale: energy cost term magnitude relative to +1.
Q5. Chemsys split vs prototype grouping vs DOI-year lookup (Crossref, 23k DOIs).
Q6. Is A-Lab "Partial" a success or a failure for AUC? ARROWS3 "trace"?
Q7. What is the 90-second demo story and which single chart leads?
Q8. What to cut if only 3 hours remain.
