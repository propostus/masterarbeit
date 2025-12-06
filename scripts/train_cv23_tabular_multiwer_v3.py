# scripts/train_cv23_tabular_multiwer_v3.py
# ------------------------------------------------------
# Trainiert LightGBM & CatBoost auf CV23-balanced (nur SigMOS + WavLM)
# Targets: wer_tiny, wer_base, wer_small
# Split: group-basiert nach client_id
# Optimiert auf R², reportet zusätzlich RMSE, MAE, CCC
# Speichert:
#   - LightGBM-Modelle: LGBM_v3_<target>.pkl
#   - CatBoost-Modelle: CatBoost_v3_<target>.pkl
#   - feature_cols_v3.txt (Feature-Reihenfolge)
#   - train_metrics_cv23_v3.csv
# ------------------------------------------------------

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
    """CCC gemäß Lin (1989)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom == 0:
        return 0.0
    return (2 * cov) / denom


def train_and_eval_model(X_train, X_test, y_train, y_test, model, model_name, target_name, out_dir):
    """Trainiert ein Modell und berechnet Metriken auf dem Test-Set."""
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    ccc = concordance_correlation_coefficient(y_test, preds)

    model_path = os.path.join(out_dir, f"{model_name}_v3_{target_name}.pkl")
    joblib.dump(model, model_path)

    return {
        "model": f"{model_name}_v3",
        "target": target_name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "ccc": ccc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM & CatBoost auf CV23-balanced (nur SigMOS+WavLM, multi-WER, v3)."
    )
    parser.add_argument("--dataset", required=True, help="Pfad zur CV23-balanced-CSV (SigMOS+WavLM+Meta+WER)")
    parser.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis für Modelle und Metriken")
    parser.add_argument("--test_size", type=float, default=0.2, help="Testanteil für GroupShuffleSplit")
    parser.add_argument("--random_state", type=int, default=42, help="Random-State für GroupShuffleSplit")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Lade Datensatz von: {args.dataset}")
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Datensatz: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Targets
    targets = ["wer_tiny", "wer_base", "wer_small"]

    # Spalten, die NICHT als Features verwendet werden sollen
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

    # Feature-Spalten: alles, was numerisch ist und nicht ausgeschlossen wurde
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    print(f"Anzahl Feature-Spalten: {len(feature_cols)}")

    # Nur Zeilen behalten, bei denen alle Targets definiert sind
    before = len(df)
    df = df.dropna(subset=targets)
    after = len(df)
    if after < before:
        print(f"Droppe {before - after} Zeilen mit NaN in Targets. Verbleibend: {after}")

    # Gruppen für GroupShuffleSplit (client_id, falls vorhanden)
    if "client_id" in df.columns:
        groups = df["client_id"].astype(str)
    else:
        # Fallback: nach filename gruppieren
        print("Warnung: Spalte 'client_id' fehlt, nutze 'filename' als Gruppe.")
        groups = df["filename"].astype(str)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_idx, test_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")
    print(f"Unique groups (Train/Test): {groups.iloc[train_idx].nunique()} / {groups.iloc[test_idx].nunique()}")

    # Features/Targets als Arrays
    X_train = train_df[feature_cols].astype(np.float32).values
    X_test = test_df[feature_cols].astype(np.float32).values

    results = []

    for target in targets:
        print(f"\n=== Training für Zielvariable: {target} ===")

        y_train = train_df[target].astype(np.float32).values
        y_test = test_df[target].astype(np.float32).values

        # LightGBM-Konfiguration (bewährt)
        lgbm = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )

        # CatBoost-Konfiguration (bewährt)
        cat = CatBoostRegressor(
            iterations=400,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            verbose=False,
            random_state=42,
        )

        res_lgb = train_and_eval_model(
            X_train, X_test, y_train, y_test, lgbm, "LightGBM", target, args.out_dir
        )
        res_cat = train_and_eval_model(
            X_train, X_test, y_train, y_test, cat, "CatBoost", target, args.out_dir
        )

        results.extend([res_lgb, res_cat])

    # Feature-Liste speichern (für Unseen-Evaluation)
    feat_path = os.path.join(args.out_dir, "feature_cols_v3.txt")
    with open(feat_path, "w") as f:
        for c in feature_cols:
            f.write(c + "\n")
    print(f"\nFeature-Liste gespeichert unter: {feat_path}")

    # Metriken speichern
    results_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "train_metrics_cv23_v3.csv")
    results_df.to_csv(out_csv, index=False)

    print("\n=== Training abgeschlossen (v3, nur SigMOS+WavLM) ===")
    print(results_df)
    print(f"\nGesamtübersicht gespeichert unter: {out_csv}")


if __name__ == "__main__":
    main()