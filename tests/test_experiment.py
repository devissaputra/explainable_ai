from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.run_experiment import clean_features, normalize_target, build_pipeline

def test_repository_is_research_bundle():
    root=Path(__file__).resolve().parents[1]
    for p in ["README.md","RESEARCH_BUNDLE.md","DATA.md","REPRODUCIBILITY.md",
              "src/run_experiment.py","paper/paper.md",".github/workflows/ci.yml"]:
        assert (root/p).exists(), p

def test_target_normalization_handles_periods():
    assert normalize_target([">50K", "<=50K.", " >50K. "]).tolist()==[1,0,1]
    with pytest.raises(ValueError):
        normalize_target(["unknown"])

def test_question_mark_becomes_missing():
    X=pd.DataFrame({"workclass":[" ? ","Private"],"age":[20,30]})
    out=clean_features(X)
    assert pd.isna(out.loc[0,"workclass"])

def test_mixed_pipeline_fits():
    X=pd.DataFrame({"age":[20,30,40,50],"job":["a","b","a","b"]})
    y=np.array([0,1,0,1])
    model=build_pipeline(X)
    model.fit(X,y)
    assert model.predict_proba(X).shape==(4,2)
