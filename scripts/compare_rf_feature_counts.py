# scripts/compare_rf_feature_counts.py
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


def evaluate_rf(dataset_path, n_splits=5):
    df = pd.read_csv(dataset_path)
    if "wer" not in df.columns:
        raise ValueError(f"Dataset {dataset_path} hat keine 'wer'-Spalte")

    X = df.drop(columns=["filename", "wer"], errors="ignore").select_dtypes(include=[np.number])
    y = df["wer"].values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ))
    ])

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_train_score=False
    )

    return {
        "dataset": os.path.basename(dataset_path).replace(".csv", ""),
        "features": X.shape[1],
        "r2_mean": np.mean(scores["test_r2"]),
        "r2_std": np.std(scores["test_r2"]),
        "mae_mean": -np.mean(scores["test_neg_mean_absolute_error"]),
        "mae_std": np.std(-scores["test_neg_mean_absolute_error"]),
        "rmse_mean": -np.mean(scores["test_neg_root_mean_squared_error"]),
        "rmse_std": np.std(-scores["test_neg_root_mean_squared_error"]),
    }


def compare_rf(datasets, out_csv, n_splits=5):
    results = []
    for ds in datasets:
        print(f"==> Evaluating RF on {ds}")
        res = evaluate_rf(ds, n_splits=n_splits)
        results.append(res)
        print(f"  R2={res['r2_mean']:.4f}, MAE={res['mae_mean']:.4f}, RMSE={res['rmse_mean']:.4f}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nErgebnisse gespeichert in: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Liste von Dataset-CSV-Dateien")
    parser.add_argument("--out_csv", type=str, required=True, help="Pfad für Ergebnis-Tabelle")
    parser.add_argument("--n_splits", type=int, default=5, help="Anzahl CV-Splits")
    args = parser.parse_args()

    compare_rf(args.datasets, args.out_csv, n_splits=args.n_splits)