# Explainable AI: Model Dependence and Feature-Governance Study

[![CI](https://github.com/devissaputra/explainable_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/explainable_ai/actions/workflows/ci.yml)
[![Empirical Study](https://github.com/devissaputra/explainable_ai/actions/workflows/empirical.yml/badge.svg)](https://github.com/devissaputra/explainable_ai/actions/workflows/empirical.yml)

**Research Bundle · AI Engineering · explanation stability and feature governance**

This repository studies how global feature explanations change across model families and feature-governance choices on the **UCI Adult** dataset. The objective is not to declare a model fair or to treat feature importance as causal evidence. The objective is to measure predictive dependence, explanation stability, performance sensitivity, subgroup behavior, and error patterns under a reproducible protocol.

## Research questions

1. Which raw features does each fitted model depend on for held-out ROC-AUC?
2. How stable are top permutation-importance features across repeated shuffles?
3. Does the explanation profile differ between logistic regression and random forest?
4. How does excluding `race` and `sex` change held-out performance and explanations?
5. What subgroup and error patterns remain even after those attributes are excluded?

## Real dataset

The study uses **UCI Adult / Census Income**, dataset 2. UCI reports 48,842 observations and 14 predictors. The target is whether annual income exceeds USD 50K. UCI reports a CC BY 4.0 license and DOI `10.24432/C5XW20`.

The runner downloads the canonical UCI archive, combines `adult.data` and `adult.test`, normalizes the target, cleans documented missing values, and records the archive SHA-256 used for each empirical run.

## Controlled comparisons

Two model families are evaluated:

- logistic regression;
- random forest.

Each family is fitted under two feature conditions:

- **all_features**: the available UCI predictors including `race` and `sex`;
- **protected_excluded**: the same design with `race` and `sex` removed before fitting.

The protected-excluded condition is a feature-governance sensitivity analysis, **not a fairness certificate**. Proxy information and structural patterns can remain.

## Evaluation and robustness

- primary stratified 75/25 split: seed 42;
- repeated splits: seeds 13, 29, 42, 73, 101;
- dummy-prior ROC-AUC baseline;
- held-out ROC-AUC for both model families and both feature conditions;
- held-out permutation importance using ROC-AUC;
- 16 permutation repeats;
- top-8 feature frequency and mean pairwise top-8 Jaccard stability;
- paired protected-exclusion ROC-AUC deltas across repeated splits;
- descriptive bootstrap interval without p-value claims;
- subgroup diagnostics by `race` and `sex` on the held-out sample;
- confusion-matrix and high-confidence error analysis.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python src/run_experiment.py
```

The full empirical run writes `results/metrics.json`, `results/summary.md`, explanation figures under `results/figures/`, and `paper/results.md`.

## Interpretation boundary

Permutation importance estimates the loss in a chosen predictive metric when a feature is shuffled in a particular fitted model and sample. It is not causal attribution, moral relevance, legal permissibility, or a universal ranking of social determinants. Correlation and proxy structure can redistribute importance. Subgroup diagnostics are descriptive and are deliberately not converted into a fairness verdict.

## Professor review path

`README.md` → `DATA.md` → `src/run_experiment.py` → `results/summary.md` → `results/metrics.json` → `RESEARCH_BUNDLE.md` → `ETHICS.md` → `paper/paper.md`.
