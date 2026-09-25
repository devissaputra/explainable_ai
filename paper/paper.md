# Held-Out Permutation Importance and Protected-Feature Governance on UCI Adult

## Status
Research-bundle manuscript scaffold. Numerical results are produced by the current real-data runner.

## Questions
Which raw features drive held-out ROC-AUC in a random-forest Adult classifier, and how does the explanation/performance profile change when race and sex are excluded from prediction?

## Design
A fixed stratified holdout is used. Preprocessing is train-only. Permutation importance is computed on the untouched test set with ROC-AUC as the scoring function and repeated shuffles for uncertainty.

## Interpretation
Permutation importance is model dependence, not causal attribution. Protected-feature exclusion is a feature-governance comparison, not proof of fairness because proxies, structural inequality and measurement choices remain.

## Data
UCI Adult, dataset 2, DOI 10.24432/C5XW20, CC BY 4.0.
