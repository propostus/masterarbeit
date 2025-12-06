# scripts/train_cv23_balanced_multiwer.py
# ---------------------------------------------
# Trainiert LightGBM & CatBoost auf
# merged_sigmos_wavlm_cv23_balanced_multiwer.csv
# Ziele: wer_tiny, wer_base, wer_small
# - GroupSplit nach client_id (kein Speaker-Leakage)
# - Speichert Modelle + Feature-Liste + Metriken
# ---------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib


def concordance_correlation_coefficient(y_true, y_pred):
    """Berechnet den Concordance Correlation Coefficient (CCC)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)
    return ccc


def train_and_eval_model(X_train, X_test, y_train, y_test, model, model_name, target_name, out_dir):
    """Trainiert ein Modell und berechnet R², RMSE, MAE, CCC auf dem Test-Set."""
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    ccc = concordance_correlation_coefficient(y_test, preds)

    model_path = os.path.join(out_dir, f"{model_name}_{target_name}.pkl")
    joblib.dump(model, model_path)

    return {
        "model": model_name,
        "target": target_name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "ccc": ccc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM & CatBoost auf CV23-balanced (multi-WER).")
    parser.add_argument("--dataset", required=True, help="Pfad zur CV23-balanced CSV (merged_sigmos_wavlm_cv23_balanced_multiwer.csv)")
    parser.add_argument("--out_dir", required=True, help="Verzeichnis zum Speichern der Modelle und Metriken")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Lade Datensatz: {args.dataset}")
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Form: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    targets = ["wer_tiny", "wer_base", "wer_small"]

    # Zeilen mit fehlenden Targets entfernen
    before_rows = df.shape[0]
    df = df.dropna(subset=targets)
    after_rows = df.shape[0]
    if after_rows < before_rows:
        print(f"Warnung: {before_rows - after_rows} Zeilen wegen fehlenden WER-Werten entfernt.")
    print(f"Verbleibende Zeilen nach Target-Cleanup: {after_rows}")

    # Feature-Spalten bestimmen:
    # Exclude IDs, Text und Targets
    exclude_cols = [
        "filename",
        "client_id",
        "age",
        "gender",
        "sentence",
        "wer_tiny",
        "wer_base",
        "wer_small",
        "reference",
        "hypothesis",
    ]
    exclude_cols = [c for c in exclude_cols if c in df.columns]

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    print(f"Anzahl Feature-Spalten: {len(feature_cols)}")

    # GroupSplit nach client_id (falls vorhanden), sonst fallback auf filename
    if "client_id" in df.columns:
        groups = df["client_id"]
        print("Gruppierung für Split: client_id")
    else:
        groups = df["filename"].str.replace(r"\.(wav|mp3|flac)$", "", regex=True)
        print("Warnung: client_id nicht vorhanden, Gruppierung über filename-Basisname.")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train-Samples: {train_df.shape[0]}, Test-Samples: {test_df.shape[0]}")
    if "client_id" in df.columns:
        print(f"Unique Speaker (Train/Test): {train_df['client_id'].nunique()} / {test_df['client_id'].nunique()}")

    # Features und Targets extrahieren
    X_train_full = train_df[feature_cols].astype(np.float32).values
    X_test_full = test_df[feature_cols].astype(np.float32).values

    results = []

    for target in targets:
        print(f"\n=== Training für Zielvariable: {target} ===")

        y_train = train_df[target].astype(np.float32).values
        y_test = test_df[target].astype(np.float32).values

        # LightGBM
        lgbm = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )

        res_lgb = train_and_eval_model(
            X_train_full, X_test_full, y_train, y_test,
            lgbm, "LightGBM", target, args.out_dir
        )

        # CatBoost
        cat = CatBoostRegressor(
            iterations=400,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            verbose=0,
            random_state=42,
        )

        res_cat = train_and_eval_model(
            X_train_full, X_test_full, y_train, y_test,
            cat, "CatBoost", target, args.out_dir
        )

        results.extend([res_lgb, res_cat])

    # Ergebnisse speichern
    results_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "train_metrics_cv23_balanced_multiwer.csv")
    results_df.to_csv(out_csv, index=False)

    print("\n=== Training abgeschlossen (CV23-balanced) ===")
    print(results_df)
    print(f"Metriken gespeichert unter: {out_csv}")

    # Feature-Liste speichern, damit Unseen-Evaluation exakt die gleichen Features nutzt
    feat_path = os.path.join(args.out_dir, "feature_cols_cv23_balanced.txt")
    with open(feat_path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(col + "\n")
    print(f"Feature-Liste gespeichert unter: {feat_path}")


if __name__ == "__main__":
    main()