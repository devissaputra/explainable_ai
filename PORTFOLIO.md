# Portfolio Summary

## Explainable AI with Permutation Importance

I train a 450-tree Random Forest on the Wisconsin Diagnostic Breast Cancer dataset, then use repeated permutation importance on held-out data to inspect which features the fitted model depends on.

### Images

![Project overview](assets/01_cover.svg)

![Explainability pipeline](assets/02_data_pipeline.svg)

![Permutation importance](assets/03_data_or_model.svg)

![Model discrimination](assets/04_evaluation_or_results.svg)

**Key result:** held-out ROC-AUC was 0.9945. The largest measured permutation effects came from worst texture and worst smoothness.

The importance values describe model dependence, not causal effects.
