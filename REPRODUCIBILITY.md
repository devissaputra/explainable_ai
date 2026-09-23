# Reproducing the experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The train/test split, Random Forest, and permutation procedure use seed 42. Feature importance is computed on the held-out test set with 16 repeats.

The crucial scoring choice is explicit:

```python
permutation_importance(..., scoring="roc_auc")
```

This aligns the explanation with the model's reported discrimination metric.

Outputs:

- `results/metrics.json`
- `results/figures/permutation_importance.png`
- `results/figures/roc_curve.png`

CI uses two permutation repeats for smoke testing so it exercises the real code path without paying the full runtime cost.
