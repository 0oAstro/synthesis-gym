# Arena rubric (picker and cross-judge only; candidates do not see this)
Score each criterion 1-5 with one line of evidence (section reference).
1. Real MDP: history changes future transitions through hidden state; partial observability is explicit and costed. No step is equivalent to an independent pull.
2. Grounded transitions: each part of the transition model is labelled physics / learned (data + labels named) / assumption, and the design calibrates against ARROWS3 intermediates and A-Lab with a named metric. Chemist critique (docs/debate/r1-chemist.md) respected.
3. Evaluation honesty: headline metrics independent of the optimised model; leakage, power (N~200, paired, Wilson), re-run baselines, and the reward-hack cases are handled.
4. Sim-to-real slot: the real lab plugs in through the same interface, and sim-to-real gap is a defined per-task number.
5. Buildability tonight: a v0 that the verified stack can implement in ~5 hours, with concrete types, signatures, files; small API, junior-readable.
6. Standard-ness: task registry, versioned benchmark ids, baselines, leaderboard-ready outputs; a stranger could adopt it.
