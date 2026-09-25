# Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python src/run_experiment.py
```

Frozen defaults:
- UCI dataset 2
- seed 42
- test fraction 0.25
- Random Forest trees 450
- permutation repeats 16
- permutation scoring: ROC-AUC

Unit tests are offline. The empirical runner requires UCI network access.

The previous Wisconsin breast-cancer metrics are not valid evidence for this bundle and were removed.
