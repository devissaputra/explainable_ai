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
- archive SHA-256: `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb`;
- grouped split method: shuffled `StratifiedGroupKFold(n_splits=4)`, first fold;
- grouping key: exact predictor vector after documented string/missing-value cleaning;
- primary seed 42;
- repeated seeds 13, 29, 42, 73, 101;
- model families: logistic regression and random forest;
- feature conditions: all predictors vs race/sex excluded;
- permutation repeats: 16;
- explanation sample cap: 4,000 held-out observations;
- top-k stability set size: 8.

## Source and split integrity

The runner rejects a source archive whose SHA-256 differs from the frozen value. Exact duplicate predictor groups are kept together so the same predictor vector cannot occur in both train and test partitions.

## Outputs

A full run generates `results/metrics.json`, `results/summary.md`, four permutation-importance figures, `paper/results.md`, and `paper/results.tex`.

## CI boundary

Unit tests are offline and cover archive parsing, source-hash validation, target normalization, grouped-split leakage protection, model fitting, subgroup denominators and error analysis. The separate empirical workflow performs the networked UCI run and regenerates evidence.

## Statistical interpretation

Repeated grouped holdouts reuse observations and are not independent replications. Bootstrap intervals over paired split-level feature-exclusion deltas are descriptive robustness summaries; no inferential p-value is claimed.

## Manuscript build

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

The LaTeX manuscript imports generated `results.tex`; numerical values should not be manually transcribed.
