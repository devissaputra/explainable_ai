# Explainable AI Research Bundle

[![CI](https://github.com/devissaputra/explainable_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/explainable_ai/actions/workflows/ci.yml)

**Research Bundle · AI Engineering · explanation stability and feature-governance audit**

This repository studies global model explanations on the **UCI Adult** dataset. It replaces the earlier breast-cancer demonstration with a real, socially consequential benchmark where explanation claims require stronger methodological discipline.

## Research questions

1. Which raw features does a fitted random-forest classifier depend on for held-out ROC-AUC?
2. How stable are the top permutation-importance features across repeated shuffles?
3. What changes when the protected attributes `race` and `sex` are excluded from the predictive feature set?

This is an explanation study, not a causal analysis and not a fairness certification.

## Real dataset

**UCI Adult / Census Income**, dataset 2.

- 48,842 records
- 14 predictors
- target: whether annual income exceeds USD 50K
- missing values are documented by UCI
- UCI license: CC BY 4.0
- DOI: 10.24432/C5XW20

The empirical runner fetches the dataset from UCI through `ucimlrepo`.

## Frozen comparison

Two models use the same algorithm, split and evaluation procedure:

- **all_features**: uses the available Adult predictors, including race and sex;
- **protected_excluded**: removes race and sex before fitting.

For each condition:

1. fixed stratified 75/25 holdout split with seed 42;
2. train-only preprocessing for missing values and categorical encoding;
3. Random Forest with 450 trees;
4. held-out ROC-AUC;
5. held-out permutation importance scored by ROC-AUC;
6. 16 permutation repeats with mean and standard deviation retained.

The point is not to declare the protected-excluded model “fair.” Excluding protected attributes does not remove proxy information or historical bias.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

Outputs are written to `results/metrics.json` and `results/figures/`.

## What makes this a Research Bundle

- externally sourced real data;
- an explicit explanation question;
- controlled feature-governance comparison;
- held-out explanation computation;
- uncertainty across repeated permutations;
- leakage-aware preprocessing;
- tests and CI;
- data/ethics/reproducibility documentation;
- paper-ready research scaffold.

See [RESEARCH_BUNDLE.md](RESEARCH_BUNDLE.md).

## Interpretation boundary

Permutation importance measures **dependence of this fitted model on a feature for this metric and sample**. It does not establish causality, moral relevance, legal permissibility, or an immutable ranking of social determinants. Correlated and proxy features can redistribute measured importance.

## Professor review path

`README.md` → `DATA.md` → `src/run_experiment.py` → `tests/` → `ETHICS.md` → `paper/paper.md` → generated results.
