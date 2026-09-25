# Ethics, Fairness Boundaries and Responsible Interpretation

UCI Adult contains sensitive social and socioeconomic information. A technically accurate classifier or explanation does not establish that income prediction is an appropriate basis for decisions about people.

## Protected-feature exclusion is not fairness

Removing `race` and `sex` does not remove correlated proxies, historical inequity, selection effects or measurement choices. This repository therefore treats exclusion as a feature-governance sensitivity analysis rather than a fairness intervention or certification.

## Explanation limits

Permutation importance measures how much a fitted model's held-out ROC-AUC changes when a feature is shuffled. It does not show that the feature causes income, deserves intervention, or is ethically legitimate to use. Correlated features can split or transfer measured importance.

## Subgroup diagnostics

The study reports descriptive subgroup ROC-AUC, TPR and FPR with group and class denominators where sample size is adequate. These values are not collapsed into a single fairness score or verdict. Any real decision context would require a context-specific legal, ethical and stakeholder analysis.

## Duplicate-aware evaluation

Exact duplicate predictor profiles are kept on one side of each train/test split. This prevents identical observed predictor vectors from inflating held-out evidence merely because one copy was seen during training. Grouping duplicates does not solve broader dependence, sampling bias or census-weight interpretation.

## Benchmark boundary

The study combines the distributed Adult train and test files for repeated grouped robustness analysis. It therefore does not claim direct comparability with results that use UCI's original fixed split.

## Deployment boundary

A deployment study would require a justified use case, contemporary and representative data, measurement review, subgroup uncertainty, calibration, proxy analysis, privacy safeguards, appeals and human oversight. This repository does not authorize automated employment, lending, benefits, admissions or eligibility decisions.
