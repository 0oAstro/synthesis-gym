# Synthesis Gym: original brief (from planning chat, 2026-11-02)

Goal: RL environment ("gym") for inorganic solid-state materials synthesis. An agent proposes
synthesis steps (precursors, temperature, time, atmosphere); a learned simulator ("oracle")
returns next state + reward. Pitch: "Every materials gym is a simulator. Ours is calibrated
against a real furnace." Tonight: sim tier only, trained on public text-mined recipes, plus a
real-lab queue stub. Demo metric: agent success rate on held-out real recipes, per round.
Deadline: YC application demo, a few hours.

Priorities:
- P0 data loader (Ceder text-mined solid-state recipes, Kononova et al. Sci Data 2019,
  CederGroupHub/text-mined-synthesis_public): target, precursors, ops (T, time, atmosphere).
- P0 oracle v0: P(recipe works | target, precursors, T, time, atm) with uncertainty.
  GBDT bootstrap ensemble (5-10), composition + process + thermo features (MP reaction energy,
  Tamman ratio), stoichiometry balance flag. Split by target formula. ROC-AUC + calibration.
  Pseudo-negatives by perturbation (swap precursor, shift T +-400C, drop precursor, wrong atm):
  PU-style weak labels, documented honestly.
- P0 Gymnasium env: obs = target + history + best_p; action dict = precursors (vocab), T [200,1700],
  time [0.5,72], atmosphere enum, or stop. Bernoulli(p) success. Reward +1 success, -0.05/step,
  -0.0002*T*t energy, -0.5 invalid. Real-lab hook stub.
- P0 agents: Random, Heuristic (frequent precursors, Tamman T, 12h, air), Claude (strict JSON,
  retry once, fallback heuristic; optional retrieval of 5 similar recipes).
- P0 eval: N=50 held-out targets; sim success + "real-recipe match" (same precursor set, T within
  +-150C); rounds with oracle retraining on agent episodes (sim-only loop). PNG charts + metrics.json.
- P1 terminal demo script; P2 lab queue stub; P3 AlchemyBench, MACE-MP-0 features, Streamlit.
Honesty: no real-lab claims; oracle is "v0 world model", not a foundation model; weak labels stated.

Environment facts: Linux arm64 (aarch64) EC2, 8 cores, 30 GiB. Python via uv, mise for tools.
No ANTHROPIC_API_KEY: Claude runs on AWS Bedrock via instance role (AnthropicBedrock client).
MP_API_KEY is unset. Model IDs: claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5, claude-fable-5-1.
Scratch space: /var/tmp (not /tmp, which is a 2 GB tmpfs).

Bedrock (verified 2026-11-02 with boto3 converse, instance role, AWS_PROFILE unset):
region us-east-1, model ids `us.anthropic.claude-opus-5-5` and `us.anthropic.claude-fable-5-1`.
