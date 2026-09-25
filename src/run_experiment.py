from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
UCI_DATASET_ID = 2
PROTECTED = ("race", "sex")

def normalize_target(raw) -> pd.Series:
    s = pd.Series(raw).astype(str).str.strip().str.rstrip(".")
    y = s.map({">50K": 1, "<=50K": 0})
    if y.isna().any():
        raise ValueError(f"Unexpected Adult target labels: {sorted(s[y.isna()].unique())}")
    return y.astype(int)

def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    frame = X.copy()
    for column in frame.select_dtypes(include=["object", "category"]).columns:
        frame[column] = frame[column].astype("object")
        frame[column] = frame[column].replace(r"^\s*\?\s*$", np.nan, regex=True)
        frame[column] = frame[column].map(lambda v: v.strip() if isinstance(v, str) else v)
    return frame

def load_real_data():
    ds = fetch_ucirepo(id=UCI_DATASET_ID)
    X = clean_features(ds.data.features)
    target = ds.data.targets
    y = normalize_target(target.iloc[:, 0] if hasattr(target, "iloc") else target)
    return X, y

def build_pipeline(X: pd.DataFrame, seed: int = SEED):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = RandomForestClassifier(
        n_estimators=450, random_state=seed, n_jobs=-1,
        class_weight="balanced_subsample"
    )
    return Pipeline([("preprocess", pre), ("model", model)])

def explain(model, X_test, y_test, n_repeats=16, seed=SEED):
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc",
        n_repeats=n_repeats, random_state=seed, n_jobs=-1
    )
    rows = [
        {
            "feature": str(X_test.columns[i]),
            "mean_roc_auc_drop": float(result.importances_mean[i]),
            "std_roc_auc_drop": float(result.importances_std[i]),
        }
        for i in np.argsort(result.importances_mean)[::-1]
    ]
    return rows

def run_condition(name, X_train, X_test, y_train, y_test, n_repeats=16, seed=SEED):
    model = build_pipeline(X_train, seed)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    return {
        "condition": name,
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "features": X_train.columns.tolist(),
        "permutation_importance": explain(model, X_test, y_test, n_repeats, seed),
    }

def run_experiment(results_dir: str | Path="results", n_repeats: int=16, seed: int=SEED, make_plots: bool=True):
    X, y = load_real_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=seed, stratify=y
    )
    lower_map = {c.lower(): c for c in X.columns}
    protected_actual = [lower_map[p] for p in PROTECTED if p in lower_map]
    X_train_reduced = X_train.drop(columns=protected_actual)
    X_test_reduced = X_test.drop(columns=protected_actual)

    all_features = run_condition("all_features", X_train, X_test, y_train, y_test, n_repeats, seed)
    protected_excluded = run_condition(
        "protected_excluded", X_train_reduced, X_test_reduced, y_train, y_test, n_repeats, seed
    )

    results = {
        "research_bundle": True,
        "dataset": {"name":"UCI Adult","uci_id":UCI_DATASET_ID,"doi":"10.24432/C5XW20",
                    "n_samples":int(len(X)),"positive_rate":float(y.mean())},
        "seed": int(seed), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "protected_attributes": protected_actual,
        "conditions": {"all_features":all_features, "protected_excluded":protected_excluded},
        "environment": {"python":platform.python_version(),"numpy":np.__version__,
                        "pandas":pd.__version__,"scikit_learn":sklearn.__version__},
    }

    out=Path(results_dir)
    out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    if make_plots:
        figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
        for key, condition in results["conditions"].items():
            top=condition["permutation_importance"][:12][::-1]
            plt.figure(figsize=(9,6))
            plt.barh([r["feature"] for r in top],[r["mean_roc_auc_drop"] for r in top],
                     xerr=[r["std_roc_auc_drop"] for r in top])
            plt.xlabel("Mean decrease in held-out ROC-AUC")
            plt.title(f"UCI Adult permutation importance: {key}")
            plt.tight_layout()
            plt.savefig(figdir/f"permutation_importance_{key}.png",dpi=160)
            plt.close()
    return results

if __name__=="__main__":
    print(json.dumps(run_experiment(),indent=2))
