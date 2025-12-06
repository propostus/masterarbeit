# scripts/train_clean_augmented_multiwer.py
# ------------------------------------------------------
# Trainiert LightGBM & CatBoost auf Clean + Augmented Datasets
# mit GroupSplit, mehreren Targets (WER Tiny/Base/Small)
# und Evaluation (R², RMSE, MAE, CCC)
# ------------------------------------------------------

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib


# ------------------------------------------------------
# Hilfsfunktion: Concordance Correlation Coefficient (CCC)
# ------------------------------------------------------
def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc


# ------------------------------------------------------
# Training + Evaluation pro Modell
# ------------------------------------------------------
def train_and_eval_model(X_train, X_test, y_train, y_test, model, model_name, target_name, out_dir):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    ccc = concordance_correlation_coefficient(y_test, preds)

    joblib.dump(model, os.path.join(out_dir, f"{model_name}_{target_name}.pkl"))

    return {"model": model_name, "target": target_name, "r2": r2, "rmse": rmse, "mae": mae, "ccc": ccc}


# ------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train LightGBM & CatBoost on Clean+Augmented dataset (multi-WER)")
    parser.add_argument("--dataset", required=True, help="Pfad zur kombinierten Clean+Augmented CSV")
    parser.add_argument("--out_dir", required=True, help="Verzeichnis zum Speichern der Modelle und Metriken")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"=== Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten ===")

    # ------------------------------------------------------
    # Fehlende WER-Werte entfernen
    # ------------------------------------------------------
    target_cols = ["wer_tiny", "wer_base", "wer_small"]
    before = len(df)
    df = df.dropna(subset=target_cols)
    after = len(df)
    print(f"→ Entfernt {before - after} Zeilen mit fehlenden WER-Werten (verbleibend: {after})")

    # ------------------------------------------------------
    # Feature-Spalten bestimmen
    # ------------------------------------------------------
    exclude_cols = [
        "filename", "group_id", "source",
        "wer_tiny", "wer_base", "wer_small",
        "reference", "hypothesis"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    # ------------------------------------------------------
    # Gruppierter Split (kein Leakage zwischen clean/aug)
    # ------------------------------------------------------
    if "group_id" not in df.columns:
        df["group_id"] = df["filename"].str.replace(r"\.(wav|mp3|flac)$", "", regex=True)
        df["group_id"] = df["group_id"].str.replace("_aug$", "", regex=True)

    groups = df["group_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    print(f"\n=== Datensplits ===")
    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")
    print(f"Unique groups (Train/Test): {train_df['group_id'].nunique()} / {test_df['group_id'].nunique()}")

    # ------------------------------------------------------
    # Training für jede Zielvariable
    # ------------------------------------------------------
    results = []
    targets = ["wer_tiny", "wer_base", "wer_small"]

    for target in targets:
        print(f"\n=== Training für Zielvariable: {target} ===")

        X_train = train_df[feature_cols].astype(np.float32).values
        y_train = train_df[target].astype(np.float32).values
        X_test = test_df[feature_cols].astype(np.float32).values
        y_test = test_df[target].astype(np.float32).values

        # Modelle definieren
        lgbm = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        cat = CatBoostRegressor(
            iterations=400,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            verbose=0,
            random_state=42,
        )

        # Training & Evaluation
        res_lgb = train_and_eval_model(X_train, X_test, y_train, y_test, lgbm, "LightGBM", target, args.out_dir)
        res_cat = train_and_eval_model(X_train, X_test, y_train, y_test, cat, "CatBoost", target, args.out_dir)

        results.extend([res_lgb, res_cat])

    # ------------------------------------------------------
    # Ergebnisse speichern
    # ------------------------------------------------------
    results_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "train_metrics_clean_augmented.csv")
    results_df.to_csv(out_csv, index=False)

    print("\n=== Fertig! ===")
    print(results_df)
    print(f"\nMetriken gespeichert unter: {out_csv}")


if __name__ == "__main__":
    main()