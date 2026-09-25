from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.run_experiment import (
    _extract_adult_files,
    _parse_adult,
    build_model,
    clean_features,
    error_analysis,
    normalize_target,
    subgroup_audit,
)


def test_repository_is_research_bundle():
    root = Path(__file__).resolve().parents[1]
    for p in [
        "README.md", "RESEARCH_BUNDLE.md", "DATA.md", "REPRODUCIBILITY.md", "ETHICS.md",
        "src/run_experiment.py", "paper/paper.md", ".github/workflows/ci.yml", ".github/workflows/empirical.yml",
    ]:
        assert (root / p).exists(), p


def test_target_normalization_handles_periods():
    assert normalize_target([">50K", "<=50K.", " >50K. "]).tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        normalize_target(["unknown"])


def test_question_mark_becomes_missing():
    X = pd.DataFrame({"workclass": [" ? ", "Private"], "age": [20, 30]})
    out = clean_features(X)
    assert pd.isna(out.loc[0, "workclass"])


def test_mixed_models_fit():
    X = pd.DataFrame({"age": [20, 30, 40, 50, 60, 70], "job": ["a", "b", "a", "b", "a", "b"]})
    y = np.array([0, 1, 0, 1, 0, 1])
    for family in ("logistic", "random_forest"):
        model = build_model(X, family, 42)
        model.fit(X, y)
        assert model.predict_proba(X).shape == (6, 2)


def test_error_analysis_counts():
    m = error_analysis([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8])
    assert m["fp"] == 1 and m["fn"] == 1


def test_subgroup_audit_respects_minimum():
    y = [0, 1, 0, 1, 0, 1]
    p = [.1, .8, .2, .7, .3, .9]
    groups = pd.Series(["A", "A", "A", "B", "B", "B"])
    out = subgroup_audit(y, p, groups, min_n=3)
    assert set(out) == {"A", "B"}


def test_parse_combined_archive():
    train = b"20, Private,1, HS-grad,9,Never-married, Sales,Not-in-family, White, Male,0,0,40, United-States, <=50K\n"
    test = b"| header\n30, Private,2, Bachelors,13,Never-married, Tech-support,Not-in-family, Black, Female,0,0,40, United-States, >50K.\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("adult.data", train)
        z.writestr("adult.test", test)
    a, b = _extract_adult_files(buf.getvalue())
    assert len(_parse_adult(a)) == 1
    assert len(_parse_adult(b, True)) == 1
