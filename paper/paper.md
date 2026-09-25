# Model Dependence, Explanation Stability and Protected-Feature Governance on UCI Adult

## Abstract

Global feature-importance plots are often presented as if they were stable properties of a dataset. This study instead treats explanations as model- and sample-dependent empirical objects. Logistic regression and random forest are evaluated on UCI Adult with and without race and sex. Held-out permutation importance, repeated permutation stability, repeated train/test splits, subgroup diagnostics and error analysis are used to distinguish predictive dependence from causal or fairness claims.

## Research questions

Which raw features support held-out discrimination in each model family? How stable are top features under permutation repeats? How do performance and explanations change when race and sex are excluded? What subgroup and error patterns remain?

## Data and design

The study uses UCI Adult dataset 2. A fixed stratified 75/25 primary split uses seed 42, with four additional fixed seeds for robustness. All preprocessing is train-only.

## Models and conditions

Logistic regression and random forest are each fitted under all-feature and protected-excluded conditions. A class-prior dummy serves as a minimal discrimination baseline.

## Explanation analysis

Permutation importance is evaluated on held-out data with ROC-AUC scoring and 16 repeats. The study records each feature's mean and standard deviation of ROC-AUC drop, top-8 frequency across repeats, and mean pairwise Jaccard similarity of top-8 sets.

## Governance and subgroup analysis

Race and sex exclusion is treated as a controlled sensitivity analysis, not a fairness certification. Both feature conditions are audited descriptively across held-out race and sex groups where sample size is adequate.

## Error analysis

False positives, false negatives, overall error rate and high-confidence errors are reported for the primary split.

## Results

Generated numerical evidence is written to `paper/results.md` and `results/metrics.json`; this methods file deliberately avoids hand-entered performance claims.

## Limitations

The dataset is historical, socioeconomic variables are entangled, explanations are model-dependent, repeated holdouts are dependent, and subgroup summaries do not resolve normative fairness questions.
