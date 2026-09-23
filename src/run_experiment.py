from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


SEED = 42


def load_split(seed: int = SEED):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def fit_model(X_train, y_train, seed: int = SEED):
    model = RandomForestClassifier(
        n_estimators=450,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def explain_model(
    model,
    X_test,
    y_test,
    n_repeats: int = 16,
    seed: int = SEED,
):
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )
    order = np.argsort(importance.importances_mean)[::-1]
    rows = []
    for index in order:
        rows.append(
            {
                "feature": str(X_test.columns[index]),
                "mean_importance": float(importance.importances_mean[index]),
                "std_importance": float(importance.importances_std[index]),
            }
        )
    return rows


def run_experiment(
    results_dir: str | Path = "results",
    n_repeats: int = 16,
    seed: int = SEED,
    make_plots: bool = True,
):
    X_train, X_test, y_train, y_test = load_split(seed)
    model = fit_model(X_train, y_train, seed)
    probability = model.predict_proba(X_test)[:, 1]
    roc_auc = float(roc_auc_score(y_test, probability))
    importance_rows = explain_model(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        seed=seed,
    )

    results = {
        "seed": int(seed),
        "n_repeats": int(n_repeats),
        "roc_auc": roc_auc,
        "importance_scoring": "roc_auc",
        "top_features": importance_rows[:10],
    }

    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    if make_plots:
        figures = output / "figures"
        figures.mkdir(parents=True, exist_ok=True)

        top = importance_rows[:10][::-1]
        plt.figure(figsize=(9, 6))
        plt.barh(
            [row["feature"] for row in top],
            [row["mean_importance"] for row in top],
            xerr=[row["std_importance"] for row in top],
        )
        plt.xlabel("Mean decrease in held-out ROC-AUC")
        plt.title("Permutation importance with 16 repeats")
        plt.tight_layout()
        plt.savefig(figures / "permutation_importance.png", dpi=150)
        plt.close()

        fpr, tpr, _ = roc_curve(y_test, probability)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Held-out model discrimination")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "roc_curve.png", dpi=150)
        plt.close()

    return results


def main() -> None:
    print(json.dumps(run_experiment(), indent=2))


if __name__ == "__main__":
    main()
