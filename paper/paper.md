# Explainable AI with Permutation Importance

## Question

After fitting a strong nonlinear classifier, which input features does the model rely on most on held-out data?

## Data

I use the Wisconsin Diagnostic Breast Cancer dataset with a stratified 75/25 train/test split.

## Method

The classifier is a Random Forest with 450 trees and `random_state=42`.

I calculate class probabilities for ROC-AUC, then run scikit-learn permutation importance on the held-out set with 16 repeats.

Permutation importance shuffles one feature at a time and measures how much the model's score changes.

## Results

The held-out ROC-AUC is 0.9945.

The five largest mean permutation effects are:

| Feature | Mean importance |
|---|---:|
| worst texture | 0.00393 |
| worst smoothness | 0.00219 |
| worst concavity | 0.00087 |
| mean concave points | 0.00087 |
| worst concave points | 0.00044 |

## Interpretation

The model depends most strongly on the features near the top of the table in this fitted run.

That statement is narrower than saying those features cause the diagnosis. Permutation importance explains model dependence. It does not establish biological causality.

Correlated predictors are another complication: if two features carry similar information, shuffling only one may have a small effect because the other can partly replace it.

## Limitations

This is a global explanation of one fitted model on one split. It does not explain an individual prediction, and it does not test whether the importance ranking is stable across retraining.

A stronger study would compare permutation importance with SHAP, group correlated features, and report stability across several random seeds.

## Reproduce

```bash
pip install -r requirements.txt
python src/run_experiment.py
```
