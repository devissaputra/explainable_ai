# Calculation guide

## Question and evidence

Do explanations change with the model and feature policy?

UCI Adult: 48,842 records; exact duplicate predictor profiles remain within the same partition.

**Status:** RECORDED EXTERNAL-DATA STUDY | full experiment not rerun in this review.

## Design

Compare logistic regression and random forest, with and without race and sex; permute raw features on held-out observations.

## Calculation and interpretation

`Importance(j) = AUC(original) - mean AUC(permuted feature j).`

Permutation importance measures dependence of a fitted predictor. Correlated predictors can redistribute importance, and shuffled combinations may be unrealistic. Removing protected attributes does not remove their proxies or establish fairness.

## Evidence table

Selected recorded values (units and context shown). Full precision below is for traceability, not a claim of measurement precision.

| Quantity | Value | Unit / meaning | JSON path |
|---|---:|---|---|
| logistic / all_features | 0.9063071122512927 | ROC-AUC ↑ | `primary.models.logistic.all_features.roc_auc` |
| logistic / protected_excluded | 0.9051812271219376 | ROC-AUC ↑ | `primary.models.logistic.protected_excluded.roc_auc` |
| random_forest / all_features | 0.9173915316754124 | ROC-AUC ↑ | `primary.models.random_forest.all_features.roc_auc` |
| random_forest / protected_excluded | 0.9167075479846082 | ROC-AUC ↑ | `primary.models.random_forest.protected_excluded.roc_auc` |

Source: [results/metrics.json](results/metrics.json). Values resolve directly from this file when figures are regenerated.

On the primary grouped holdout, random-forest ROC-AUC is 0.9174 with all features and 0.9167 after excluding race and sex; logistic regression scores 0.9063 and 0.9052. Marital status leads the permutation rankings, but its importance differs markedly between models. Subgroup diagnostics remain necessary after exclusion, so the study supports inspection of model dependence rather than causal attribution or a fairness verdict.

## Verification performed in this review

The existing suite requires unavailable dependencies; no full-suite pass is claimed. The complete data/model experiment was not rerun in this review. Stored empirical results were inspected, not independently reproduced from raw data.

The figure-generation check verifies agreement between the selected source values and SVGs. It does not validate the raw dataset, fitted model, identification assumptions, or external generalization.

```bash
python scripts/build_review_figures.py
python scripts/build_review_figures.py --check
```

## Implementation map

Follow these functions to inspect each transformation. Validation helpers and private functions remain visible in the linked modules.

| Function | Purpose / documented behavior |
|---|---|
| [`validate_archive_hash`](src/run_experiment.py#L52) | Inspect the explicit implementation and its callers. |
| [`normalize_target`](src/run_experiment.py#L86) | Inspect the explicit implementation and its callers. |
| [`clean_features`](src/run_experiment.py#L94) | Inspect the explicit implementation and its callers. |
| [`load_real_data`](src/run_experiment.py#L102) | Inspect the explicit implementation and its callers. |
| [`predictor_group_ids`](src/run_experiment.py#L147) | Stable row-group ids so exact predictor duplicates cannot cross train/test. |
| [`grouped_stratified_split`](src/run_experiment.py#L156) | Inspect the explicit implementation and its callers. |
| [`build_preprocessor`](src/run_experiment.py#L165) | Inspect the explicit implementation and its callers. |
| [`build_model`](src/run_experiment.py#L180) | Inspect the explicit implementation and its callers. |
| [`explanation_subset`](src/run_experiment.py#L196) | Inspect the explicit implementation and its callers. |
| [`permutation_explanation`](src/run_experiment.py#L205) | Inspect the explicit implementation and its callers. |
| [`error_analysis`](src/run_experiment.py#L245) | Inspect the explicit implementation and its callers. |
| [`subgroup_audit`](src/run_experiment.py#L259) | Inspect the explicit implementation and its callers. |
| [`fit_condition`](src/run_experiment.py#L286) | Inspect the explicit implementation and its callers. |
| [`paired_condition_deltas`](src/run_experiment.py#L300) | Inspect the explicit implementation and its callers. |
| [`build_results_latex`](src/run_experiment.py#L322) | Inspect the explicit implementation and its callers. |
| [`run_experiment`](src/run_experiment.py#L373) | Inspect the explicit implementation and its callers. |
| [`write_summary`](src/run_experiment.py#L474) | Inspect the explicit implementation and its callers. |
| [`write_figures`](src/run_experiment.py#L539) | Inspect the explicit implementation and its callers. |
| [`main`](src/run_experiment.py#L558) | Inspect the explicit implementation and its callers. |

## What remains before a stronger research claim

Permutation importance measures dependence of a fitted predictor. Correlated predictors can redistribute importance, and shuffled combinations may be unrealistic. Removing protected attributes does not remove their proxies or establish fairness. A successful software test is not validation of a scientific construct. New experiments should state their split unit, comparator, outcome, uncertainty procedure and failure criteria before examining final test results.
