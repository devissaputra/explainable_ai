# Explainable AI: Model Dependence and Feature-Governance Study

This bundle studies how global feature explanations change across model families and feature-governance choices on the Adult income benchmark. It combines held-out performance, permutation-importance stability, subgroup behavior, feature exclusion, and error analysis while keeping predictive dependence separate from causal or fairness claims.

On the primary grouped holdout, random-forest ROC-AUC is 0.9174 with all features and 0.9167 after excluding race and sex; logistic regression scores 0.9063 and 0.9052. Marital status leads the permutation rankings, but its importance differs markedly between models. Subgroup diagnostics remain necessary after exclusion, so the study supports inspection of model dependence rather than causal attribution or a fairness verdict.

## Start here

- [Calculations, evidence and verification scope](CALCULATIONS.md)
- [Figure sources and exact numerical paths](docs/figure_spec.json)
- [Working paper](paper/paper.md)
- [Data and provenance](DATA.md)

![Study question, data, design and interpretation](assets/review_overview.svg)

![Defined calculation and source-linked evidence](assets/review_calculations.svg)

**Review scope:** The existing suite requires unavailable dependencies; no full-suite pass is claimed. The complete data/model experiment was not rerun in this review. Stored empirical results were inspected, not independently reproduced from raw data.

## Detailed project documentation

[![CI](https://github.com/devissaputra/explainable_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/explainable_ai/actions/workflows/ci.yml)
[![Empirical Study](https://github.com/devissaputra/explainable_ai/actions/workflows/empirical.yml/badge.svg)](https://github.com/devissaputra/explainable_ai/actions/workflows/empirical.yml)

**Research Bundle · AI Engineering · explanation stability and feature governance**

This repository studies how global feature explanations change across model families and feature-governance choices on the **UCI Adult** income-classification benchmark. The objective is not to declare a model fair or to treat feature importance as causal evidence. The study measures predictive dependence, explanation stability, performance sensitivity, subgroup behavior and error patterns under a reproducible, duplicate-aware holdout protocol.

## Research questions

1. Which raw predictors does each fitted model depend on for held-out ROC-AUC?
2. How stable are top permutation-importance features across repeated shuffles?
3. How does the explanation profile differ between logistic regression and random forest?
4. How does excluding `race` and `sex` change held-out discrimination and explanations?
5. What subgroup and error patterns remain after those attributes are excluded?

## Real dataset

The study uses **UCI Adult / Census Income**, dataset 2. UCI reports 48,842 observations and 14 predictors. The binary target is whether annual income exceeds USD 50K. UCI reports DOI `10.24432/C5XW20` and a CC BY 4.0 license.

The runner downloads the canonical UCI archive and accepts only the frozen SHA-256:

`7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb`

A different archive fails fast rather than silently changing the study.

## Split and leakage discipline

UCI distributes `adult.data` and `adult.test`. This study combines them because its objective is repeated robustness analysis rather than reproduction of the original fixed benchmark split.

Before each split, identical predictor rows receive the same group identifier. A shuffled `StratifiedGroupKFold(n_splits=4)` produces an approximately 75/25 train/test partition while keeping exact duplicate predictor vectors entirely within one side of the split. This prevents an identical predictor profile from appearing in both training and evaluation data.

All imputation, scaling and categorical encoding are learned only inside the training pipeline.

## Controlled comparisons

Two model families are evaluated: logistic regression and random forest.

Each family is fitted under two conditions:

- **all_features**: includes the available predictors, including `race` and `sex`;
- **protected_excluded**: removes `race` and `sex` before fitting.

Protected-feature exclusion is a governance sensitivity analysis, **not a fairness certificate**. Correlated proxies and structural patterns can remain.

## Evaluation and robustness

- primary grouped split: seed 42;
- repeated grouped splits: seeds 13, 29, 42, 73, 101;
- class-prior ROC-AUC baseline;
- held-out ROC-AUC for both model families and feature conditions;
- held-out raw-feature permutation importance using ROC-AUC;
- 16 permutation repeats;
- maximum 4,000 held-out observations for explanation computation;
- top-8 frequency and mean pairwise top-8 Jaccard stability;
- paired protected-exclusion ROC-AUC deltas across repeated splits;
- descriptive bootstrap intervals without p-value claims;
- descriptive subgroup audits by `race` and `sex`, including group/class denominators;
- confusion counts and high-confidence error analysis.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python src/run_experiment.py
```

The full empirical run generates `results/metrics.json`, `results/summary.md`, four explanation figures, `paper/results.md`, and `paper/results.tex`.

## Interpretation boundary

Permutation importance estimates how a fitted model's held-out ROC-AUC changes when one raw feature is shuffled. It is not causal attribution, moral relevance, legal permissibility, or a universal ranking of socioeconomic determinants. Correlation and proxy structure can redistribute measured importance. Removing race and sex does not remove proxy information. Subgroup diagnostics are descriptive and are not converted into a fairness verdict.

## Professor review path

`README.md` → `DATA.md` → `src/run_experiment.py` → `results/summary.md` → `results/metrics.json` → `RESEARCH_BUNDLE.md` → `ETHICS.md` → `paper/paper.md`.
