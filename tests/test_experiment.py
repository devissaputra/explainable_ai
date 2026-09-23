from pathlib import Path

from src.run_experiment import run_experiment


def test_importance_is_scored_with_roc_auc(tmp_path):
    result = run_experiment(
        tmp_path,
        n_repeats=2,
        make_plots=False,
    )
    assert result["importance_scoring"] == "roc_auc"
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert len(result["top_features"]) == 10


def test_importance_rows_are_well_formed(tmp_path):
    result = run_experiment(
        tmp_path,
        n_repeats=2,
        make_plots=False,
    )
    for row in result["top_features"]:
        assert set(row) == {
            "feature",
            "mean_importance",
            "std_importance",
        }
        assert row["std_importance"] >= 0.0


def test_repository_structure():
    root = Path(__file__).resolve().parents[1]
    for relative_path in [
        "README.md",
        "DATA.md",
        "ETHICS.md",
        "REPRODUCIBILITY.md",
        "src/run_experiment.py",
        "paper/paper.md",
        ".github/workflows/ci.yml",
    ]:
        assert (root / relative_path).exists(), relative_path
