# Explainable AI with Permutation Importance

![Project overview](assets/01_cover.svg)

I built this project to answer a question that comes after model training: once a nonlinear classifier performs well, how can I inspect what it is relying on?

The experiment uses a Random Forest on the Wisconsin Diagnostic Breast Cancer dataset and measures feature importance by repeatedly shuffling one feature at a time on held-out data.

## Data and model

I use:

- 569 samples;
- 30 numerical features;
- a stratified 75/25 train/test split;
- a Random Forest with 450 trees;
- random state 42.

The model's held-out class probabilities are evaluated with ROC-AUC.

## How the explanation works

![Explainability pipeline](assets/02_data_pipeline.svg)

Permutation importance asks a practical question:

> How much does model performance drop when one feature is randomly shuffled?

If shuffling a feature hurts the score, the fitted model was using information carried by that feature.

I repeat each feature permutation 16 times to reduce the effect of one lucky or unlucky shuffle.

## Feature importance

![Permutation importance](assets/03_data_or_model.svg)

The largest mean importance values in the recorded run were:

| Feature | Mean importance |
|---|---:|
| worst texture | 0.00393 |
| worst smoothness | 0.00219 |
| worst concavity | 0.00087 |
| mean concave points | 0.00087 |
| worst concave points | 0.00044 |

The values are small because scikit-learn's permutation importance is measuring the change in the estimator's default score on the held-out set.

These numbers show model dependence, not causality. Correlated features can also substitute for one another, which can make an individually useful feature appear less important.

## Results

![Model discrimination](assets/04_evaluation_or_results.svg)

The recorded ROC-AUC is **0.9945** on the held-out split.

That tells me the classifier ranks the two classes very well in this experiment. It does not mean the model is ready for clinical use, and permutation importance is not an explanation of why a biological outcome occurs.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

On Windows, use `.venv\Scripts\activate`.

## Repository notes

- [DATA.md](DATA.md) explains the data source.
- [ETHICS.md](ETHICS.md) explains the limits of interpreting a medical benchmark.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) records the experiment settings.
- [paper/paper.md](paper/paper.md) contains the longer write-up.
