# Research Bundle Evidence Contract

## Identity

**Area:** AI Engineering  
**Study:** model dependence, explanation stability and protected-feature governance  
**Dataset:** UCI Adult, dataset 2

## A valid full result must record

1. canonical UCI source, DOI, license and enforced archive SHA-256;
2. sample count, feature count and positive-class prevalence;
3. exact-predictor duplicate statistics;
4. grouped train/test split method, seed, sizes and zero exact predictor-group overlap;
5. train-only missing-value handling and categorical encoding;
6. logistic-regression and random-forest model families;
7. all-feature and race/sex-excluded conditions;
8. class-prior baseline and held-out ROC-AUC;
9. held-out raw-feature permutation importance with repeat count and sample size;
10. top-k explanation frequency and pairwise-Jaccard stability;
11. paired feature-exclusion performance deltas across repeated grouped splits;
12. subgroup diagnostics for race and sex with group and class denominators;
13. overall error analysis and high-confidence errors;
14. software environment and generated figures;
15. generated Markdown and LaTeX results.

## Leakage contract

Exact duplicate predictor vectors must not cross training and test partitions. Preprocessing is learned only from training data. Permutation importance is computed only on held-out observations.

## Statistical boundary

Repeated holdouts reuse observations and are not independent replications. Bootstrap intervals over split-level deltas are descriptive robustness summaries.

## Non-claims

Protected-feature exclusion is not proof of fairness. Permutation importance is not causal attribution. Historical income classification is not assumed to be a legitimate deployment objective. Subgroup metrics are descriptive rather than an overall fairness ranking.
