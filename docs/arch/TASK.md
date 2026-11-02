# Arena task: architecture for Synthesis Gym as a real MDP

Produce an architecture document for **Synthesis Gym: the reference RL environment for
inorganic solid-state materials synthesis**. It must be a genuine sequential decision problem,
not a contextual bandit, and it must ship a v0 in about 5 hours on the verified stack.

## Grounding (read all)
- /home/ec2-user/Developer/rl-achroma/docs/brief.md: original brief and environment facts.
- /home/ec2-user/Developer/rl-achroma/docs/SPEC.md: draft spec v1 (superseded where it conflicts with decisions below).
- /home/ec2-user/Developer/rl-achroma/docs/debate/r1-chemist.md, r1-ml.md, r1-yc.md: critiques with measured numbers.
- Data on disk (read-only): /home/ec2-user/Developer/rl-achroma/data/raw/ holds
  solid-state_dataset_20200713.json.xz (Ceder recipes; operations carry per-step heating T/time/atmosphere,
  and records have reaction strings and per-material element dicts), mp_nostruct_20181018.json.gz
  (83,989 MP formation energies, eV/atom, load with matminer `load_dataset` or gzip+json),
  YBCO_Exp.json / NTMO_Exp.json / LTOPO_Exp.json (ARROWS3: per precursor set, per temperature,
  the XRD-observed product phases and weight fractions, i.e. real intermediates and failures),
  alab_synthesis_results.csv (A-Lab outcomes; Failed rows have no recipe).
  Loader reference: /var/tmp/synthgym-research/data/loader_snippet.py.
- Stack verified on this host: Python 3.13, gymnasium 1.4, pymatgen, matminer, lightgbm, sklearn,
  numpy/pandas, pydantic, rich, matplotlib, anthropic[bedrock] (Claude Opus 5.5 via Bedrock).
  No Materials Project API key; no GPU; 8 CPU cores.

## Decisions already made by the owner (do not reopen)
- The bandit critique is accepted: history must change future transitions (real path dependence).
- Cut the self-training "rounds" loop on sim labels; a retraining loop is allowed only over real results.
- Leakage fixes: vocab, frequency tables and retrieval index from the training split only; chemsys
  split with subset/superset chemsys excluded across the split; report with and without retrieval.
- Evaluation needs power: N about 200 held-out targets, paired comparisons, Wilson intervals; baselines
  are re-run on our split (not quoted from papers); temperature baseline includes a median predictor.
- Reward: no Bernoulli sampling as the default sim reward; the energy/time cost must be
  decision-relevant but bounded so "stop immediately" or "30 min anneal" is not optimal.
- Pseudo-negatives that are trivially separable or chemically wrong (melt flag, standalone
  carbonate decomposition, atmosphere negatives on non-reducible cations) are out.
- Headline result is independent of the model the agent optimises: literature match on held-out
  targets and real-outcome AUC on ARROWS3 (stratified by temperature) and A-Lab (target-level).

## What the document must contain
1. State model: what the true (hidden) state is, what the agent observes, and why history matters.
   Consider a phase-assemblage state (which phases exist, in what fractions, after each step),
   characterization as an action with cost (partial observability), regrind/reheat, and ARROWS3-style
   intermediates that consume driving force. Argue for your choice against at least two alternatives.
2. Transition model: how the next state is computed. Say exactly which parts are physics
   (thermodynamics from formation energies), which are learned (and from what data, with what labels),
   and which are assumptions. Say how it is calibrated against the ARROWS3 intermediate data and A-Lab.
3. Action space and observation space as concrete Gymnasium spaces, plus the dict/JSON form an LLM agent uses.
4. Reward and termination.
5. Task suite: how targets become tasks (difficulty tiers, train/test splits, versioned benchmark ids
   like `SynthGym-v0/...`), and the registry API.
6. Real-lab tier: how a real furnace plugs in through the same interface, and how the sim-to-real gap is
   measured per task.
7. Evaluation protocol: metrics, baselines (random, heuristic, retrieval, Claude), N, statistics.
8. Module layout with key types and function signatures (Python), small API surface, junior-readable.
9. A v0 cut that ships in about 5 hours, and what is explicitly deferred to v1/v2.
10. Risks: where the sim is most likely wrong and how the design makes that visible.

Write for an engineer who will implement it tonight. Be concrete: formulas, shapes, file names.
Max 2,500 words. End with a short rationale naming the alternatives you considered and rejected.
