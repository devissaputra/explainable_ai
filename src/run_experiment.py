# Calculation reading guide: ../CALCULATIONS.md (repository root).
# Importance(j) = AUC(original) - mean AUC(permuted feature j).
# Permutation importance measures dependence of a fitted predictor. Correlated predictors can redistribute importance, and shuffled combinations may be unrealistic. Removing protected attributes does not remove their proxies or establish fairness.

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PRIMARY_SEED = 42
REPEATED_SEEDS = (13, 29, 42, 73, 101)
UCI_DATASET_ID = 2
DATA_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
DATA_DOI = "10.24432/C5XW20"
DATA_LICENSE = "CC BY 4.0"
PROTECTED = ("race", "sex")
PERMUTATION_REPEATS = 16
EXPLANATION_SAMPLE = 4000
TOP_K = 8
EXPECTED_ARCHIVE_SHA256 = "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
EXPECTED_ROWS = 48842
EXPECTED_FEATURES = 14
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]


def validate_archive_hash(payload: bytes, expected_sha256: str = EXPECTED_ARCHIVE_SHA256) -> str:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Unexpected UCI Adult archive SHA-256: {actual}; expected {expected_sha256}. "
            "The frozen research protocol requires the exact validated archive bytes."
        )
    return actual


def _extract_adult_files(payload: bytes) -> tuple[bytes, bytes]:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
        data_name = next((n for n in names if n.endswith("adult.data")), None)
        test_name = next((n for n in names if n.endswith("adult.test")), None)
        if not data_name or not test_name:
            raise FileNotFoundError("adult.data and adult.test must exist in UCI Adult archive")
        return zf.read(data_name), zf.read(test_name)


def _parse_adult(raw: bytes, test_file: bool = False) -> pd.DataFrame:
    text = raw.decode("utf-8", errors="replace")
    if test_file:
        text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("|"))
    frame = pd.read_csv(
        io.StringIO(text),
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values=["?", " ?"],
    )
    return frame.dropna(how="all").reset_index(drop=True)


def normalize_target(raw) -> pd.Series:
    s = pd.Series(raw).astype(str).str.strip().str.rstrip(".")
    y = s.map({">50K": 1, "<=50K": 0})
    if y.isna().any():
        raise ValueError(f"Unexpected Adult target labels: {sorted(s[y.isna()].unique())}")
    return y.astype(int)


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    frame = X.copy()
    for column in frame.select_dtypes(include=["object", "category"]).columns:
        frame[column] = frame[column].map(lambda v: v.strip() if isinstance(v, str) else v)
        frame[column] = frame[column].replace(r"^\s*\?\s*$", np.nan, regex=True)
    return frame


def load_real_data(data_path: str | Path | None = None, cache_dir: str | Path = "data/cache"):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / "adult.zip"
    if data_path is not None:
        payload = Path(data_path).read_bytes()
        source = f"local:{data_path}"
    elif cached.exists():
        payload = cached.read_bytes()
        source = f"cache:{cached}"
    else:
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:
            payload = response.read()
        cached.write_bytes(payload)
        source = DATA_URL

    archive_sha256 = validate_archive_hash(payload)
    train_raw, test_raw = _extract_adult_files(payload)
    frame = pd.concat([_parse_adult(train_raw), _parse_adult(test_raw, True)], ignore_index=True)
    y = normalize_target(frame.pop("income"))
    X = clean_features(frame)
    if len(X) != EXPECTED_ROWS or X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Unexpected UCI Adult shape: {X.shape}; expected "
            f"({EXPECTED_ROWS}, {EXPECTED_FEATURES}) predictors"
        )
    group_ids = predictor_group_ids(X)
    group_counts = pd.Series(group_ids).value_counts()
    return X, y, {
        "name": "UCI Adult",
        "uci_id": UCI_DATASET_ID,
        "doi": DATA_DOI,
        "license": DATA_LICENSE,
        "source": source,
        "canonical_source": DATA_URL,
        "archive_sha256": archive_sha256,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "predictor_unique_groups": int(pd.Series(group_ids).nunique()),
        "duplicate_predictor_rows": int(len(X) - pd.Series(group_ids).nunique()),
        "largest_predictor_group": int(group_counts.max()),
    }


def predictor_group_ids(X: pd.DataFrame) -> np.ndarray:
    """Stable row-group ids so exact predictor duplicates cannot cross train/test."""
    canonical = X.copy()
    for column in canonical.columns:
        if canonical[column].dtype == object:
            canonical[column] = canonical[column].fillna("<missing>").astype(str)
    return pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype=np.uint64)


def grouped_stratified_split(X: pd.DataFrame, y: pd.Series, seed: int):
    groups = predictor_group_ids(X)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    if set(groups[train_idx]).intersection(set(groups[test_idx])):
        raise RuntimeError("Exact predictor group leaked across train/test")
    return train_idx, test_idx, groups


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def build_model(X: pd.DataFrame, family: str, seed: int):
    if family == "logistic":
        estimator = LogisticRegression(max_iter=3000, random_state=seed, class_weight="balanced")
    elif family == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([("preprocess", build_preprocessor(X)), ("model", estimator)])


def explanation_subset(X: pd.DataFrame, y: pd.Series, seed: int, max_n: int = EXPLANATION_SAMPLE):
    if len(X) <= max_n:
        return X, y
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=max_n, random_state=seed, stratify=y
    )
    return X_small, y_small


def permutation_explanation(model, X_test, y_test, seed: int, n_repeats: int = PERMUTATION_REPEATS):
    X_exp, y_exp = explanation_subset(X_test, y_test, seed)
    result = permutation_importance(
        model,
        X_exp,
        y_exp,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)[::-1]
    rows = [{
        "feature": str(X_exp.columns[i]),
        "mean_roc_auc_drop": float(result.importances_mean[i]),
        "std_roc_auc_drop": float(result.importances_std[i]),
        "top_k_frequency": float(np.mean([
            i in np.argsort(result.importances[:, r])[::-1][:TOP_K]
            for r in range(result.importances.shape[1])
        ])),
    } for i in order]

    top_sets = [
        set(np.argsort(result.importances[:, r])[::-1][:TOP_K].tolist())
        for r in range(result.importances.shape[1])
    ]
    jaccards = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            union = top_sets[i] | top_sets[j]
            jaccards.append(len(top_sets[i] & top_sets[j]) / len(union) if union else 1.0)
    return {
        "n_explanation_samples": int(len(X_exp)),
        "n_repeats": int(n_repeats),
        "top_k": TOP_K,
        "mean_pairwise_top_k_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
        "ranking": rows,
    }


def error_analysis(y_true, probability) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    wrong = pred != y
    confidence = np.where(pred == 1, p, 1 - p)
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "error_rate": float(wrong.mean()),
        "high_confidence_errors_ge_0_80": int((wrong & (confidence >= .80)).sum()),
    }


def subgroup_audit(y_true, probability, groups: pd.Series, min_n: int = 50) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= .5).astype(int)
    out = {}
    clean_groups = groups.fillna("<missing>").astype(str).to_numpy()
    for group in sorted(np.unique(clean_groups)):
        mask = clean_groups == group
        if mask.sum() < min_n:
            continue
        yg, pg, pr = y[mask], p[mask], pred[mask]
        normal = yg == 0
        positive = yg == 1
        positive_n = int(positive.sum())
        negative_n = int(normal.sum())
        out[group] = {
            "n": int(mask.sum()),
            "positive_n": positive_n,
            "negative_n": negative_n,
            "positive_rate": float(yg.mean()),
            "roc_auc": float(roc_auc_score(yg, pg)) if positive_n and negative_n else None,
            "tpr_at_0_5": float(((pr == 1) & positive).sum() / positive_n) if positive_n else None,
            "fpr_at_0_5": float(((pr == 1) & normal).sum() / negative_n) if negative_n else None,
        }
    return out


def fit_condition(X_train, X_test, y_train, y_test, family: str, seed: int, explain: bool = False):
    model = build_model(X_train, family, seed)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    result = {
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "error_analysis": error_analysis(y_test, probability),
        "probability": probability,
    }
    if explain:
        result["permutation_importance"] = permutation_explanation(model, X_test, y_test, seed)
    return result


def paired_condition_deltas(repeated: list[dict]) -> dict:
    rng = np.random.default_rng(20260925)
    out = {}
    for family in ("logistic", "random_forest"):
        deltas = np.asarray([
            row[family]["protected_excluded"] - row[family]["all_features"]
            for row in repeated
        ], dtype=float)
        boot = []
        for _ in range(4000):
            idx = rng.integers(0, len(deltas), size=len(deltas))
            boot.append(float(deltas[idx].mean()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[family] = {
            "mean_roc_auc_delta_excluded_minus_all": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "bootstrap_95_interval": [float(lo), float(hi)],
            "note": "Descriptive paired-split interval; repeated holdouts are not independent replications.",
        }
    return out


def build_results_latex(results: dict) -> str:
    bs = "\\"
    row_end = bs + bs
    lines = [
        f"{bs}section{{Generated empirical results}}",
        "This section is generated by \\texttt{src/run\_experiment.py}; numerical values should not be hand-edited.",
        "",
        f"Dataset: UCI Adult, $n={results['dataset']['n_samples']:,}$, "
        f"{results['dataset']['n_features']} predictors, positive prevalence "
        f"{results['dataset']['positive_rate']:.4f}.",
        "",
        f"{bs}begin{{table}}[htbp]",
        f"{bs}centering",
        f"{bs}small",
        f"{bs}begin{{tabular}}{{lrrrr}}",
        f"{bs}toprule",
        "Model & All-feature AUC & Excluded AUC & Top-8 stability all & Top-8 stability excluded " + row_end,
        f"{bs}midrule",
    ]
    for family, d in results["primary"]["models"].items():
        a, e = d["all_features"], d["protected_excluded"]
        lines.append(
            f"{family.replace('_', ' ').title()} & {a['roc_auc']:.4f} & {e['roc_auc']:.4f} & "
            f"{a['permutation_importance']['mean_pairwise_top_k_jaccard']:.4f} & "
            f"{e['permutation_importance']['mean_pairwise_top_k_jaccard']:.4f} " + row_end
        )
    lines += [
        f"{bs}bottomrule",
        f"{bs}end{{tabular}}",
        f"{bs}caption{{Primary grouped holdout results. Exact duplicate predictor rows are kept within one split partition.}}",
        f"{bs}label{{tab:primary-results}}",
        f"{bs}end{{table}}",
        "",
        f"{bs}paragraph{{Feature-exclusion sensitivity.}}",
    ]
    for family, delta in results["paired_feature_exclusion_deltas"].items():
        lo, hi = delta["bootstrap_95_interval"]
        lines.append(
            f"{family.replace('_', ' ').title()}: mean AUC change "
            f"{delta['mean_roc_auc_delta_excluded_minus_all']:.4f}, descriptive 95\% interval "
            f"[{lo:.4f}, {hi:.4f}]."
        )
    lines += [
        "",
        f"{bs}paragraph{{Interpretation.}}",
        "Removing race and sex changes discrimination only slightly under this protocol, but this does not establish fairness: correlated proxies, measurement choices, and subgroup error differences can remain.",
        "",
    ]
    return "\n".join(lines)


def run_experiment(results_dir: str | Path = "results", data_path: str | Path | None = None, quick: bool = False):
    X, y, dataset = load_real_data(data_path=data_path)
    protected_actual = [c for c in PROTECTED if c in X.columns]
    seeds = (PRIMARY_SEED,) if quick else REPEATED_SEEDS
    repeated = []
    primary_detail = None

    for seed in seeds:
        train_idx, test_idx, groups = grouped_stratified_split(X, y, seed)
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        reduced_train = X_train.drop(columns=protected_actual)
        reduced_test = X_test.drop(columns=protected_actual)
        row = {"seed": seed}
        detail = {
            "seed": seed,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "exact_predictor_group_overlap": int(
                len(set(groups[train_idx]).intersection(set(groups[test_idx])))
            ),
            "models": {},
        }
        for family in ("logistic", "random_forest"):
            all_result = fit_condition(
                X_train, X_test, y_train, y_test, family, seed, explain=(seed == PRIMARY_SEED)
            )
            reduced_result = fit_condition(
                reduced_train, reduced_test, y_train, y_test, family, seed, explain=(seed == PRIMARY_SEED)
            )
            row[family] = {
                "all_features": all_result["roc_auc"],
                "protected_excluded": reduced_result["roc_auc"],
            }
            if seed == PRIMARY_SEED:
                detail["models"][family] = {
                    "all_features": {k: v for k, v in all_result.items() if k != "probability"},
                    "protected_excluded": {k: v for k, v in reduced_result.items() if k != "probability"},
                    "subgroup_audit_all_features": {
                        attr: subgroup_audit(y_test, all_result["probability"], X_test[attr])
                        for attr in protected_actual
                    },
                    "subgroup_audit_protected_excluded": {
                        attr: subgroup_audit(y_test, reduced_result["probability"], X_test[attr])
                        for attr in protected_actual
                    },
                }
        repeated.append(row)
        if seed == PRIMARY_SEED:
            primary_detail = detail

    dummy_train_idx, dummy_test_idx, _ = grouped_stratified_split(X, y, PRIMARY_SEED)
    dummy_train, dummy_test = X.iloc[dummy_train_idx], X.iloc[dummy_test_idx]
    dummy_y_train, dummy_y_test = y.iloc[dummy_train_idx], y.iloc[dummy_test_idx]
    dummy = DummyClassifier(strategy="prior").fit(dummy_train, dummy_y_train)
    dummy_auc = float(roc_auc_score(dummy_y_test, dummy.predict_proba(dummy_test)[:, 1]))

    results = {
        "research_bundle": True,
        "status": "quick_smoke_run" if quick else "complete",
        "dataset": dataset,
        "protocol": {
            "target_test_fraction": .25,
            "split_method": "StratifiedGroupKFold(n_splits=4), first fold; exact predictor duplicates grouped",
            "primary_seed": PRIMARY_SEED,
            "repeated_seeds": list(seeds),
            "protected_attributes": protected_actual,
            "model_families": ["logistic", "random_forest"],
            "permutation_repeats": PERMUTATION_REPEATS,
            "explanation_sample_max": EXPLANATION_SAMPLE,
            "top_k_stability": TOP_K,
        },
        "dummy_prior_roc_auc_primary": dummy_auc,
        "primary": primary_detail,
        "repeated_split_performance": repeated,
        "paired_feature_exclusion_deltas": paired_condition_deltas(repeated),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results, out / "summary.md")
    write_figures(results, out)
    Path("paper").mkdir(exist_ok=True)
    Path("paper/results.md").write_text(
        "# Results\n\n" + (out / "summary.md").read_text(encoding="utf-8").replace("# Empirical Results Summary\n\n", "", 1),
        encoding="utf-8",
    )
    Path("paper/results.tex").write_text(build_results_latex(results), encoding="utf-8")
    return results


def write_summary(results: dict, path: Path) -> None:
    p = results["primary"]
    lines = [
        "# Empirical Results Summary", "",
        "Generated by `src/run_experiment.py`; numerical results should not be edited by hand.", "",
        f"Dataset: UCI Adult, n={results['dataset']['n_samples']:,}; positive rate={results['dataset']['positive_rate']:.4f}.", "",
        "## Primary split", "",
        "| Model | All features ROC-AUC | Protected excluded ROC-AUC | Top-k stability (all) | Top-k stability (excluded) |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, d in p["models"].items():
        a, e = d["all_features"], d["protected_excluded"]
        lines.append(
            f"| {family} | {a['roc_auc']:.4f} | {e['roc_auc']:.4f} | "
            f"{a['permutation_importance']['mean_pairwise_top_k_jaccard']:.4f} | "
            f"{e['permutation_importance']['mean_pairwise_top_k_jaccard']:.4f} |"
        )
    lines += [
        "", "## Top held-out permutation features", "",
        "| Model | Condition | Top raw features by mean ROC-AUC drop |",
        "|---|---|---|",
    ]
    for family, d in p["models"].items():
        for condition in ("all_features", "protected_excluded"):
            ranking = d[condition]["permutation_importance"]["ranking"][:8]
            feature_text = ", ".join(
                f"{r['feature']} ({r['mean_roc_auc_drop']:.4f})" for r in ranking
            )
            lines.append(f"| {family} | {condition} | {feature_text} |")
    lines += [
        "", "## Frozen data and split integrity", "",
        f"- archive SHA-256: \`{results['dataset']['archive_sha256']}\`",
        f"- exact duplicate predictor rows in full dataset: {results['dataset']['duplicate_predictor_rows']}",
        f"- exact predictor-group overlap in the primary train/test split: {results['primary']['exact_predictor_group_overlap']}", "",
        "## Repeated split feature-exclusion sensitivity", "",
    ]
    for family, d in results["paired_feature_exclusion_deltas"].items():
        lo, hi = d["bootstrap_95_interval"]
        lines.append(
            f"- {family}: mean AUC Δ excluded-minus-all = {d['mean_roc_auc_delta_excluded_minus_all']:.4f}; "
            f"descriptive 95% bootstrap interval [{lo:.4f}, {hi:.4f}]"
        )
    lines += ["", "## Protected-excluded subgroup diagnostics", ""]
    for family, d in p["models"].items():
        lines += [
            f"### {family}", "",
            "| Attribute | Group | n | Positive n | Negative n | ROC-AUC | TPR @ 0.5 | FPR @ 0.5 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for attribute, groups in d["subgroup_audit_protected_excluded"].items():
            for group, metrics in groups.items():
                fmt = lambda x: "NA" if x is None else f"{x:.4f}"
                lines.append(
                    f"| {attribute} | {group} | {metrics['n']} | {metrics['positive_n']} | "
                    f"{metrics['negative_n']} | {fmt(metrics['roc_auc'])} | "
                    f"{fmt(metrics['tpr_at_0_5'])} | {fmt(metrics['fpr_at_0_5'])} |"
                )
        lines.append("")
    lines += [
        "## Interpretation guardrail", "",
        "Protected-feature exclusion is a governance sensitivity analysis, not a fairness certificate. Permutation importance measures predictive dependence of a fitted model on held-out data; it is not causal attribution. Group diagnostics are descriptive and are not converted into a fairness verdict.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figures(results: dict, results_dir: Path) -> None:
    figdir = results_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for family, family_data in results["primary"]["models"].items():
        for condition in ("all_features", "protected_excluded"):
            ranking = family_data[condition]["permutation_importance"]["ranking"][:12][::-1]
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(
                [r["feature"] for r in ranking],
                [r["mean_roc_auc_drop"] for r in ranking],
                xerr=[r["std_roc_auc_drop"] for r in ranking],
            )
            ax.set_xlabel("Mean decrease in held-out ROC-AUC")
            ax.set_title(f"UCI Adult permutation importance: {family} / {condition}")
            fig.tight_layout()
            fig.savefig(figdir / f"importance_{family}_{condition}.png", dpi=170)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UCI Adult explainability research bundle")
    parser.add_argument("--data-path", default=None, help="Optional local UCI Adult zip")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="Primary split only")
    args = parser.parse_args()
    result = run_experiment(args.results_dir, args.data_path, args.quick)
    print(json.dumps({"status": result["status"], "dataset": result["dataset"]}, indent=2))


if __name__ == "__main__":
    main()
