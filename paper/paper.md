# Explainable AI: Scientific-Style Technical Report

**Status:** reproducible portfolio report, not peer reviewed.  
**Dataset:** Wisconsin Diagnostic Breast Cancer dataset  
**Difficulty:** ★★★★

## Abstract
Train a nonlinear classifier and inspect held-out permutation importance and probability behavior. The repository emphasizes traceable data processing, reproducible implementation, task-appropriate evaluation, and explicit limitations.

## Method
1. Load real data
2. Split
3. Random forest
4. Permutation importance
5. Interpretation audit

## Evaluation
Primary metric(s): ROC-AUC / importance. Validation: stratified hold-out.

## Results
```json
{
  "roc_auc": 0.9945492662473795,
  "top_features": [
    [
      "worst texture",
      0.003933566433566425
    ],
    [
      "worst smoothness",
      0.0021853146853146807
    ],
    [
      "worst concavity",
      0.0008741258741258723
    ],
    [
      "mean concave points",
      0.0008741258741258723
    ],
    [
      "worst concave points",
      0.00043706293706293614
    ],
    [
      "worst symmetry",
      0.0
    ],
    [
      "worst compactness",
      0.0
    ],
    [
      "concave points error",
      0.0
    ],
    [
      "symmetry error",
      0.0
    ],
    [
      "fractal dimension error",
      0.0
    ]
  ]
}
```

## Limitations
association is not causality. Results on one public benchmark do not establish universal model quality.

## Reproducibility
Install `requirements.txt` and run `python src/run_experiment.py`.

## Reference
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
