# Reproducing the Experiment

Run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The train/test split is stratified with `random_state=42`.

The Random Forest uses 450 trees and `random_state=42`. Permutation importance is calculated on the held-out set with 16 repeats and the same random seed.

The script writes ROC-AUC and the ranked feature-importance values to `results/metrics.json`.

Permutation importance can move slightly across package versions or retraining choices, so record the scikit-learn version when comparing exact values.
