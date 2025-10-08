# scripts/sanity_check_rf.py
import argparse
import json
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def load_and_clean(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        wer_cols = [c for c in df.columns if c.startswith("wer_")]
        raise ValueError(f"Zielspalte '{target_col}' nicht gefunden. wer_* Spalten im File: {wer_cols}")

    # alle anderen wer_* Spalten entfernen (Leakage verhindern)
    wer_cols = [c for c in df.columns if c.startswith("wer_")]
    extra_wer = [c for c in wer_cols if c != target_col]

    # Features: alles Numerische außer filename, target, extra_wer
    drop_cols = ["filename", target_col] + extra_wer
    X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
    y = df[target_col].to_numpy()

    return df, X, y, extra_wer


def build_rf(n_estimators=600, max_features="sqrt", min_samples_leaf=1, random_state=42):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state
        ))
    ])


def cv_scores(model, X, y, n_splits=5, n_repeats=5, random_state=42):
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    mae_list, rmse_list = [], []
    for tr, te in cv.split(X, y):
        model.fit(X.iloc[tr], y[tr])
        yp = model.predict(X.iloc[te])
        mae_list.append(mean_absolute_error(y[te], yp))
        rmse_list.append(root_mean_squared_error(y[te], yp))
    return {
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2)),
        "mae_mean": float(np.mean(mae_list)),
        "mae_std": float(np.std(mae_list)),
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "folds": len(r2)
    }


def null_distribution(model, X, y, n_runs=5, n_splits=5, n_repeats=5, random_state=42):
    means = []
    for r in range(n_runs):
        y_shuf = shuffle(y, random_state=random_state + r)
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        r2 = cross_val_score(model, X, y_shuf, cv=cv, scoring="r2", n_jobs=-1)
        means.append(np.mean(r2))
    return {
        "null_mean": float(np.mean(means)),
        "null_std": float(np.std(means)),
        "runs": int(n_runs)
    }


def main():
    ap = argparse.ArgumentParser(description="Sanity-Check für RandomForest mit CV und Target-Shuffle-Baseline")
    ap.add_argument("--dataset_csv", required=True, type=str)
    ap.add_argument("--target_col", required=True, type=str, help="z.B. wer_tiny / wer_base / wer_small")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--null_runs", type=int, default=5)
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_features", type=str, default="sqrt", help='z.B. "sqrt", "log2", oder float (0-1)')
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_json", type=str, default=None, help="Optionaler Pfad für JSON-Output")
    args = ap.parse_args()

    df, X, y, extra_wer = load_and_clean(args.dataset_csv, args.target_col)

    print(f"Datei: {args.dataset_csv}")
    print(f"Samples: {len(df)} | Features (nach Cleanup): {X.shape[1]}")
    if extra_wer:
        print(f"Entfernte zusätzliche wer_* Spalten: {extra_wer}")
    else:
        print("Keine zusätzlichen wer_* Spalten gefunden.")

    rf = build_rf(
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )

    print("\n== Echte CV-Leistung ==")
    real_scores = cv_scores(rf, X, y, args.n_splits, args.n_repeats, args.random_state)
    print(f"R^2:  mean={real_scores['r2_mean']:.4f}  std={real_scores['r2_std']:.4f}")
    print(f"MAE:  mean={real_scores['mae_mean']:.4f}  std={real_scores['mae_std']:.4f}")
    print(f"RMSE: mean={real_scores['rmse_mean']:.4f}  std={real_scores['rmse_std']:.4f}")

    print("\n== Target-Shuffle-Baseline ==")
    null_scores = null_distribution(rf, X, y, args.null_runs, args.n_splits, args.n_repeats, args.random_state)
    print(f"Shuffle R^2: mean={null_scores['null_mean']:.4f}  std={null_scores['null_std']:.4f}  (runs={null_scores['runs']})")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        payload = {
            "dataset_csv": args.dataset_csv,
            "target_col": args.target_col,
            "samples": int(len(df)),
            "n_features": int(X.shape[1]),
            "removed_wer_columns": extra_wer,
            "cv": {
                "n_splits": args.n_splits,
                "n_repeats": args.n_repeats,
                **real_scores
            },
            "shuffle_baseline": null_scores,
            "rf_params": {
                "n_estimators": args.n_estimators,
                "max_features": args.max_features,
                "min_samples_leaf": args.min_samples_leaf,
                "random_state": args.random_state
            }
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nErgebnis gespeichert unter: {args.out_json}")


if __name__ == "__main__":
    main()