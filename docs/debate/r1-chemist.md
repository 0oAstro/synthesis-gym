# Round 1, chemist: the oracle rewards textbook-oxide habits, not synthesis

Numbers: `python3 -I` runs on 2026-11-02 over the Ceder clean table (20,707 rows), ARROWS3 Exp.json and the A-Lab CSV.

1. **Melt flag is wrong for half the data.** 49.7% of real recipes heat above the melting point of a precursor. Li2CO3 (723 C): 78% of its recipes exceed it; Bi2O3 79%, V2O5 75%, Sb2O3 96%, NH4H2PO4 99%. The melt is how the reaction proceeds. Negative type 2 labels the most common correct recipes as failures. Change: drop melt as a negative; keep `T/Tm` as a feature from a compound Tm table (top-100 precursors cover 87.7% of occurrences), never elemental proxies (Na metal at 98 C for Na2CO3).

2. **Carbonate flag penalises BaTiO3.** 30% of BaCO3, 22% of Li2CO3 and 21% of Na2CO3 recipes run below the listed standalone decomposition T. An acidic oxide pulls CO2 off 200 to 400 C lower. Change: remove the flag, or use reaction onsets (BaCO3 700, SrCO3 750, CaCO3 650, Li2CO3 450, Na2CO3 500 C) as a feature only.

3. **Atmosphere is 58% "unspecified".** Type 4 negatives always carry an explicit atmosphere, so the model learns "unspecified = success". A reducing negative for Al2O3, ZrO2 or MgO targets is false: stable in H2 at 1000 C. Change: type 4 only for targets with an H2-reducible cation (Cu, Ni, Co, Fe3+, Pb, Bi, Ag, Mn4+, V5+, Mo6+, W6+); negatives of an "unspecified" positive stay "unspecified".

4. **Types 1 and 3 are already hard rules.** Both break element balance, which §5 already rejects with -0.5, and 64% of rows lack a complete dE (35.5% full mp_nostruct coverage), so NaN becomes the label. Change: balance-preserving negatives: carbonate for oxide below the point-2 onset, nitrate or hydroxide above 1200 C, time cut to 1/20 at the same T.

5. **T shift is nearly trivial.** Median T is 1100 C; 5.6% of positives sit below 600 C. A -400 to -600 C shift lands where positives never are. Change: shift -200 to -300 C, and report pseudo-AUC with T removed to show how much is temperature alone.

6. **Reward hack.** The oracle is time-blind (hours summed across calcination and sintering) and the energy term is linear in time: 1100 C for 12 h costs -2.64 against +1. Optimal policy: stop at step 0, or 0.5 h at 1100 C with the commonest oxides and "unspecified". Change: cap the energy term at -0.2 per episode, log time, and the 1/20-time negative from point 4.

7. **Positive means "published", not "phase-pure".** Records state no yield, and the median chemsys has one row (4,461 chemsys), so retrieval collapses onto Li-Mn-O (293), Fe-Li-O-P (209), Ba-O-Ti (203). Change: caption "published recipe, purity unknown"; cap each chemsys at 20 training rows.

8. **A-Lab Failed rows have no recipe.** Precursors, T and duration are empty for all 15 Failed targets; every recipe is 4 h in air, so the A-Lab AUC is target-level only. Labels: Success and Success (offline) = 1, Failed = 0, Partial out of the binary AUC and into an ordinal AUC (Failed < Partial < Success). Change: name it "target feasibility AUC".

9. **ARROWS3 label is temperature in disguise.** YBCO verified runs: 600 and 700 C give 0% target in all 61; 900 C gives >=50% in 28 of 38. Label >=50% = 1, else 0; report the 900 C stratum (28 vs 10) as the recipe-level AUC. LTOPO is 56 of 58 >=50%: calibration only. `Y2(CO3)3`, `Ba2(CuO2)3`, `Y2Cu2O5` are absent from the vocab; `NaHO`, `PH9(NO2)2` need reduced-formula matching. Change: label by target weight fraction, stratify by T, normalise formulas.

10. **Q3.** Informative: balance-preserving swaps, time cut, moderate T shift, reducible-cation atmosphere. Trivial: types 1, 3, large T shift, melt. Decision: the point-4 set.

11. **Q5.** Chemsys split; remove Ba-Cu-O-Y (87 rows), Li-O-P-Ti (18) and all A-Lab chemsys from training. Decision: chemsys.

12. **Q6.** Partial and trace are failures in the binary AUC and the middle class in the ordinal one. Decision: binary on extremes, ordinal alongside.

13. **The partner's line:** "Your oracle never saw a real failure, your real failures have no recipes, and sim success is Bernoulli on perturbations you chose." Change: lead with literature match and stratified ARROWS3 AUC; sim success last.
