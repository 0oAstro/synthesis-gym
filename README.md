# Synthesis Gym

Models that learn materials synthesis from failed experiments.

Papers only publish the recipes that worked, so every public synthesis dataset is missing the
failures. We built a gym where a model runs solid-state syntheses, fails, sees why through the same
XRD a chemist would pay for, and gets better. Training on those failures beats training on clean
successes, and it turns a 3B model that could barely run an experiment into one that hits the target
on 76% of materials it has never seen.

- **SmolLM3-3B goes from 8% to 76% success** on 200 held-out targets after GRPO in the gym
  (paired p = 2e-41).
- **Learning from failures beats clean data.** On the same Qwen3-4B base and the same rollout
  budget, GRPO on successes and failures reaches 33% recipe match against the published literature.
  Fine-tuning on successes alone reaches 25% (p = 0.005).
- **The simulator tracks real labs.** It ranks real YBCO synthesis outcomes from ARROWS3 with an AUC
  of 0.79 on runs it was never fit on.

## Demo

![film](results/video/film.gif)

Full video: [results/video/film.mp4](results/video/film.mp4) (74 s).

SmolLM3-3B on Eu2Ir2O7, a material it never trained on. Before training, its own XRD shows 87%
target, and it keeps heating anyway, up to 1500 C, until the phase is gone. After GRPO, the same
model heats once at the published 1000 C, checks the XRD, stops, and ends with 100% target.

## Benchmarks

SynthGym-v0/test-200 has 200 targets from chemical systems that never appear in training. Recipe
match compares the model's precursors and temperature with the published recipe, so it never reads
the simulator.

| agent | simulated success (w >= 0.9) | recipe match vs literature | mean return |
|---|---|---|---|
| Claude Opus 5.5 | 83% | 46% | 1.17 |
| Qwen3-4B-Instruct + GRPO | 76.5% | 33% | 1.03 |
| SmolLM3-3B + GRPO | 76% | 22.5% | 1.03 |
| Qwen3-4B-Instruct base | 74.5% | 27.5% | 0.98 |
| Qwen3-4B-Instruct, clean successes only | 73.5% | 25% | 0.95 |
| heuristic (most common precursors, Tammann T) | 62% | 33% | 0.95 |
| random | 21% | 0% | 0.49 |
| SmolLM3-3B base | 8% | 5.5% | 0.12 |

![results](results/success_by_agent.png)

### SmolLM3-3B, before and after GRPO

| SmolLM3-3B (n = 200) | precursor match | recipe match | simulated success | mean return |
|---|---|---|---|---|
| base | 10.5% | 5.5% | 8% [5, 13] | 0.12 |
| + GRPO | 43.5% | 22.5% | 76% [70, 81] | 1.03 |

Every metric improves, with exact McNemar p = 3e-20 for precursor match, 2e-8 for recipe match and
2e-41 for success. The model learns to drive the lab: format errors fall from 2.64 to 0.05 per
episode. It also picks the published precursors four times as often, and that metric never touches
the simulator.

### Clean successes vs learning from failures

This is the experiment the company rests on. Both arms start from Qwen3-4B-Instruct and get the same
rollout budget.

- **Clean successes only.** We sampled 931 episodes, kept the 641 that reached the target and threw
  away the 290 failures. That is what training on published recipes amounts to. Then we fine-tuned
  on the successes (`scripts/train_rft.py`).
- **Learning from failures.** GRPO trains on every episode, failed ones included, for 60 steps
  (`scripts/train_grpo_tools.py`).

| arm (n = 200 held-out) | simulated success | recipe match | mean return |
|---|---|---|---|
| base | 74.5% | 27.5% | 0.98 |
| clean successes only | 73.5% | 25.0% | 0.95 |
| learning from failures (GRPO) | 76.5% | 33.0% | 1.03 |

Training on clean successes gave nothing over the base model. Training on failures moved recipe match
from 25% to 33% against the real literature (exact McNemar p = 0.005) and raised return
(Wilcoxon p = 0.046). The failures carry information that the clean data does not.

### Claude Opus 5.5 vs a literature heuristic

| agent (n = 200) | precursor set = literature | set + T within 150 C | T MAE (C) | simulated success | mean return |
|---|---|---|---|---|---|
| Claude Opus 5.5 | 64% [57, 70] | 46% [39, 53] | 139 | 83% [77, 88] | 1.17 |
| heuristic | 58% [52, 65] | 33% [27, 40] | 202 | 62% [56, 69] | 0.95 |
| random | 0% | 0% | 477 | 21% [16, 27] | 0.49 |

Brackets are Wilson 95% intervals. Claude beats the heuristic on recipe match (p = 0.001) and on
success (p = 2e-10). Guessing the median temperature gives a 244 C error, so Claude's 139 C is real
chemistry knowledge.

## What we built

- **A hidden crucible.** The agent adds precursors, heats under a chosen atmosphere, regrinds, pays
  for XRD and stops. It never sees the true phases, only noisy XRD it pays for, the same position a
  chemist is in.
- **Real thermodynamics.** Every heat solves a linear program for the lowest-Gibbs-energy products
  across 83,989 Materials Project phases. Gas release, oxygen exchange with the atmosphere,
  carbonate decomposition and reduction under H2 all come out of that one minimisation, with no
  hand-written rules.
- **Kinetics fit to real experiments.** A four-constant rate law, fit to ARROWS3 phase fractions,
  decides how far each reaction gets. History matters as it does in a real furnace: early
  intermediates use up driving force, and sintered powder reacts slowly until it is reground
  (`synthgym/sim.py`, `synthgym/kinetics.py`).
- **Real tasks.** 931 training targets and 362 test targets from the Ceder group's 9,628 text-mined
  training recipes, split by chemical system so test materials are new to the model.
- **A tool environment for RL.** Models act through `add_precursors`, `heat`, `regrind`, `xrd` and
  `stop`, and see each result before the next call (`synthgym/tool_env.py`). It plugs into TRL GRPO
  through `environment_factory` and evaluates on vLLM.

## What's next

The gym proves the method. The data is the business. We are starting a pilot with a materials
science professor at IIT Delhi to record every pathway a lab tries, failed ones included, and to
put trained agents on real furnaces. Every run makes the dataset that no one else has.

## Run

```sh
uv sync
uv run python scripts/fit_kinetics.py          # fits the rate law on ARROWS3 (writes models/kinetics.json)
uv run python scripts/run_eval.py random heuristic claude
uv run python scripts/demo.py --task SynthGym-v0/hard/BaFe12O19 --agent claude
uv run python scripts/train_grpo_tools.py      # GRPO in the gym
uv run python scripts/eval_tools.py --name smollm3-grpo --model HuggingFaceTB/SmolLM3-3B --lora checkpoints/grpo-tools
```

Claude runs on AWS Bedrock (`us.anthropic.claude-opus-5-5`). Raw data goes in `data/raw/`.

## Data sources

- Ceder group text-mined solid-state recipes, Kononova et al., Sci. Data 2019
  (github.com/CederGroupHub/text-mined-synthesis_public).
- ARROWS3, Szymanski et al., Nat. Commun. 2023 (github.com/njszym/ARROWS).
- A-Lab, Szymanski et al., Nature 2023.
- Materials Project formation energies through matminer `mp_nostruct_20181018`.
