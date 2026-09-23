# Explainable AI with Held-Out Permutation Importance

**Focus:** aligning the explanation metric with the evaluation metric.

A Random Forest on the Wisconsin Diagnostic Breast Cancer benchmark reaches held-out ROC-AUC 0.9945. Permutation importance is computed on the held-out set using `scoring="roc_auc"`, so each importance value directly measures how much ROC-AUC falls when a feature is shuffled.

The repository retains both mean importance and standard deviation across 16 repeats. It also makes the interpretation boundary explicit: permutation importance measures fitted-model dependence, not biological causality, and correlated features can substitute for one another.

The experiment is import-safe, reproducible, behaviorally tested, and checked by GitHub Actions.
