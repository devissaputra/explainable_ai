# Explainable AI: Model Dependence and Feature-Governance Study

This bundle studies how global feature explanations change across model families and feature-governance choices on the Adult income benchmark. It combines held-out performance, permutation-importance stability, subgroup behavior, feature exclusion, and error analysis while keeping predictive dependence separate from causal or fairness claims.

On the primary grouped holdout, random-forest ROC-AUC is 0.9174 with all features and 0.9167 after excluding race and sex; logistic regression scores 0.9063 and 0.9052. Marital status leads the permutation rankings, but its importance differs markedly between models. Subgroup diagnostics remain necessary after exclusion, so the study supports inspection of model dependence rather than causal attribution or a fairness verdict.

See [CALCULATIONS.md](CALCULATIONS.md) for evidence and verification scope.
