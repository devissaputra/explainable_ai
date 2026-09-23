# Explainable AI with Held-Out Permutation Importance

## Abstract

This experiment explains a Random Forest classifier on the Wisconsin Diagnostic Breast Cancer benchmark using held-out permutation importance. The explanation metric is explicitly aligned with the evaluation metric: both use ROC-AUC.

## Method

The model uses 450 trees and a stratified 75/25 split with seed 42. Held-out ROC-AUC is 0.9945. Each feature is permuted 16 times on the test set, and the decrease in ROC-AUC is recorded.

## Results

The largest mean ROC-AUC decreases are observed for worst area (0.00485), worst concave points (0.00376), worst perimeter (0.00375), and mean concave points (0.00217). Standard deviations across repeats are retained because several effects are small and overlapping.

## Interpretation

Permutation importance measures dependence of this fitted model on a feature under held-out shuffling. It does not establish biological causality. Correlated predictors can substitute for one another and reduce apparent individual importance.

## Limitations

A stronger study would evaluate explanation stability across splits and models, grouped or conditional permutation importance, SHAP as a comparison, calibration, and external validation.
