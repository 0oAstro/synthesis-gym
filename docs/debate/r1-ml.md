# Round 1, ML/RL evaluation critique of SPEC v1

(computed) = measured on the cleaned Ceder table, 20,707 rows, 2026-11-02.

1. **It is a contextual bandit, not an MDP.** The oracle depends on (target, action) only; history is in the observation, not the transition or reward. The optimal policy is argmax_a p(target, a) repeated; max_steps=5 is five independent pulls with a step tax. Cheap fix: call it a contextual bandit and report regret against the best vocab recipe under judge_oracle. Multi-step fix, doable tonight: state = current phase assemblage. A failed step returns predicted products (carbonate residue below decomposition T, unreacted precursor below Tamman onset; ARROWS3 Exp.json has `products` per temperature as a template), and the next step reacts those products, so "anneal hotter" and "add a precursor" become path dependent. Spec change: §5 renames to `SynthesisBandit` and drops history from obs, or adds `state.phases` fed into the next step's features.

2. **env_oracle vs judge_oracle is not independence.** Same features, negative generators and vocab, halves of one distribution: an agent that learns "T above 900 C, carbonate for alkaline earths, air" scores high on both for the same reason. Disjoint chemsys blocks memorisation, not bias. Spec change: report Spearman agreement of the two oracles on random valid actions; above 0.9, the caption calls the judge a replica and sim success a footnote.

3. **Q2, the rounds loop, is self-confirming: cut it.** Round k adds episodes labelled Bernoulli(env_oracle p) to env_oracle's own training set. The expected label is the model's own prediction, so retraining yields a sharper copy of itself; a rising `success_by_round.png` plots growing overconfidence. No information enters without the furnace. Spec change: delete rounds in §7. If a loop is wanted, round k ingests only `source="real"` rows and the chart is real-outcome AUC against rows ingested.

4. **Leakage paths.** (a) Retrieval's nearest neighbour of a held-out target is a doped sibling in a sister chemsys, and 75% of recipes within a chemsys share one precursor set (computed): retrieval is near-copy. (b) The 1,689-precursor vocab is built on all data; a rare held-out precursor is reachable only because its string leaked. (c) Ba-Ti-O and Ba-Ti-Zr-O are different chemsys and the same recipe plus ZrO2. Spec change: vocab and frequency tables from the train fold only; exclude from retrieval any training target whose element set is a subset or superset of the test target; report match with and without retrieval.

5. **N=50 cannot answer the question.** Wilson 95%: 18/50 = 36% gives [24.1%, 49.9%]; 25/50 = 50% gives [36.6%, 63.4%]; they overlap. Separating 36% from 50% (alpha 0.05, power 0.8, unpaired) needs 196 targets per arm; 36% vs 60% needs 67. He 2023's 36% is from another split and version; the per-element modal heuristic scores 52% in-sample here (computed), so rerun the baseline, do not quote it. Ceiling: 16% of recipes do not use their target's modal precursor set (computed). Spec change: N=200, paired on the same targets with McNemar; Random and Heuristic are free; Claude plain is one call per target, about 200 x 2k tokens, under 5 USD on Bedrock. Report T MAE against the 247 C MAE of predicting the median (computed).

6. **Bernoulli sampling adds variance only.** E[success] = p, so sampling widens intervals; the agent reads p and std from history anyway and optimises env_oracle p directly. Spec change: sim reward = p (or p minus lambda·std); Bernoulli only behind `stochastic=True`.

7. **Q4, reward scale.** At 1000 C x 12 h the term is 0.024: half the step tax, 2% of success, never decision-relevant. Spec change: drop it, or set the coefficient to 3.8e-6 so the 90th-percentile recipe (1475 C, 53 h, computed) costs 0.3.

8. **Atmosphere negatives are trivially separable.** 58% of positives are `unspecified` (computed), so negative type 4 teaches "any named atmosphere is suspicious". Spec change: type-4 negatives only from positives with a named atmosphere; drop `unspecified` from the one-hot.

9. **Real-outcome AUC is mostly one target.** YBCO supplies 149 of 223 real rows; the AUC measures process sensitivity on Ba-Y-Cu-O. Spec change: AUC per source and macro average.

Decisions: Q1, lead with literature match plus real-outcome AUC; sim success tertiary, agreement number printed. Q2, cut. Q4, drop the energy term or rescale to 3.8e-6.
