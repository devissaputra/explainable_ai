# Explainable AI with Held-Out Permutation Importance

[![CI](https://github.com/devissaputra/explainable_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/explainable_ai/actions/workflows/ci.yml)

![Project overview](assets/01_cover.svg)

A compact explainability study built around a simple rule:

> **The explanation metric should match the performance metric being explained.**

The classifier is evaluated with ROC-AUC, so permutation importance is also computed using **ROC-AUC loss under feature shuffling**, not the estimator's default accuracy score.

## Data and model

- Wisconsin Diagnostic Breast Cancer benchmark
- 569 observations
- 30 numerical features
- stratified 75/25 train/test split
- Random Forest with 450 trees
- seed 42

All feature importance is computed on the held-out test set.

## Explanation method

![Explainability pipeline](assets/02_data_pipeline.svg)

For each feature, permutation importance:

1. measures held-out ROC-AUC;
2. randomly shuffles one feature;
3. measures ROC-AUC again;
4. records the performance drop;
5. repeats the shuffle 16 times.

A larger positive value means the fitted classifier depends more strongly on that feature for **held-out discrimination**.

## Recorded model performance

| Metric | Result |
|---|---:|
| ROC-AUC | **0.9945** |

## Top held-out permutation importances

![Permutation importance](assets/03_data_or_model.svg)

| Feature | Mean ROC-AUC drop | Std. dev. |
|---|---:|---:|
| worst area | 0.00485 | 0.00247 |
| worst concave points | 0.00376 | 0.00203 |
| worst perimeter | 0.00375 | 0.00242 |
| mean concave points | 0.00217 | 0.00113 |
| worst texture | 0.00176 | 0.00064 |
| mean texture | 0.00106 | 0.00062 |
| worst concavity | 0.00096 | 0.00071 |
| mean concavity | 0.00086 | 0.00061 |
| worst smoothness | 0.00072 | 0.00042 |
| compactness error | 0.00052 | 0.00026 |

![Model discrimination](assets/04_evaluation_or_results.svg)

The error bars matter. Several features have small, overlapping effects, and correlated measurements can substitute for one another.

## What this explanation does not mean

Permutation importance is **model dependence**, not biological causality.

If two features contain similar information, shuffling one may appear unimportant because the model can still use the other. A feature can also look important because of associations in this specific dataset without being causal or clinically actionable.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

## Test

```bash
pip install pytest
pytest
```

The CI smoke test uses fewer permutation repeats for speed but exercises the same ROC-AUC-scored path.

## Engineering details

- import-safe experiment module
- explanation scoring explicitly set to `roc_auc`
- held-out-only feature permutation
- mean and standard deviation retained
- behavioural tests and GitHub Actions
- generated plots separated from curated SVG assets

## Limits

This is a small benchmark and one model family. A stronger XAI study would add repeated splits, grouped/correlated-feature importance, SHAP or conditional importance as a comparison, stability analysis, calibration, and external data.
