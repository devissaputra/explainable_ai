# Research Bundle Evidence Contract

## Identity

**Area:** AI Engineering  
**Study:** model dependence, explanation stability and protected-feature governance  
**Dataset:** UCI Adult, dataset 2

## A result qualifies as bundle evidence only if it records

1. canonical UCI source, DOI, license and archive SHA-256;
2. sample count, feature count and positive-class prevalence;
3. split seed, train/test sizes and repeated-split protocol;
4. train-only missing-value handling and categorical encoding;
5. logistic-regression and random-forest model families;
6. all-feature and protected-excluded conditions;
7. dummy-prior baseline and held-out ROC-AUC;
8. held-out permutation importance with repeat count;
9. top-k explanation-stability frequency and pairwise Jaccard;
10. paired protected-exclusion performance deltas across repeated splits;
11. subgroup diagnostics for race and sex;
12. overall error analysis and high-confidence errors;
13. software environment and generated figures.

## Statistical boundary

Repeated holdouts reuse observations and are not independent replications. Bootstrap intervals over paired split-level deltas are descriptive robustness summaries; no inferential p-value is claimed.

## Non-claims

Protected-feature exclusion is not proof of fairness. Permutation importance is not causal attribution. Historical income classification is not assumed to be a legitimate deployment objective. Subgroup metrics are descriptive rather than an overall fairness ranking.
