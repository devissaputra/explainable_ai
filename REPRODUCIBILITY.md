# Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python src/run_experiment.py
```

## Frozen defaults

- UCI Adult dataset 2;
- primary seed 42;
- repeated seeds 13, 29, 42, 73, 101;
- stratified test fraction 0.25;
- model families: logistic regression and random forest;
- protected-feature conditions: all features vs race/sex excluded;
- permutation repeats: 16;
- explanation sample cap: 4,000 held-out observations;
- stability set size: top 8 features.

## Data identity

The empirical manifest records the SHA-256 hash of the downloaded UCI archive. Raw data are cached under `data/cache/` and are not versioned.

## Outputs

A full run generates `results/metrics.json`, `results/summary.md`, four permutation-importance figures, and `paper/results.md`.

## CI boundary

Unit tests are offline. The separate `Empirical Study` workflow performs the networked UCI run and commits generated evidence after changes to the research runner.

## Result policy

Do not reuse metrics from the retired breast-cancer demonstration or from the earlier single-model protocol. Numerical claims for the current bundle must be taken from the generated current-protocol result files.
