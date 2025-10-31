# scripts/train_tabular_single_target.py
# Trainiert LightGBM und CatBoost auf einem einzelnen WER-Ziel (z. B. wer_tiny)
# mit gruppiertem Split, Metrik-Logging und zusätzlicher Auswertung nur für clean (snr == "clean")

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from collections import Counter
import lightgbm as lgb
from catboost import CatBoostRegressor

# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------
def evaluate(y_true, y_pred, label):
    """Berechnet R², RMSE, MAE."""
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{label}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    return {"model": label, "r2": r2, "rmse": rmse, "mae": mae}


def check_snr_distribution(train_df, val_df):
    """Prüft Verteilung der SNR-Werte."""
    print("\n=== SNR-Verteilung (Train vs. Validation) ===")
    for name, df in [("Train", train_df), ("Validation", val_df)]:
        counts = Counter(df["snr"])
        total = sum(counts.values())
        dist = {k: f"{(v/total)*100:.1f}%" for k, v in counts.items()}
        print(f"{name}: {dist}")


# ------------------------------------------------------------
# Hauptlogik
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train LightGBM & CatBoost auf einzelnes WER-Ziel")
    parser.add_argument("--dataset", required=True, help="Pfad zur CSV-Datei")
    parser.add_argument("--target", required=True, help="Zielvariable (z. B. wer_tiny)")
    parser.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== Training LightGBM & CatBoost auf {args.dataset} (Target: {args.target}) ===")

    # --------------------------------------------------------
    # Daten laden
    # --------------------------------------------------------
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    exclude_cols = ["filename", "snr", "wer_tiny", "wer_base", "wer_small"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    X = df[feature_cols].astype(np.float32).values
    y = df[args.target].astype(np.float32).values

    # --------------------------------------------------------
    # Gruppierter Split nach filename (kein Leakage über SNR)
    # --------------------------------------------------------
    print("=== Erstelle gruppierten Split nach filename (alle SNR-Versionen gemeinsam) ===")
    groups = df["filename"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in gss.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    check_snr_distribution(train_df, val_df)

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------
    print("\nTrainiere LightGBM ...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=-1,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    y_pred_lgb = lgb_model.predict(X_val)
    lgb_metrics = evaluate(y_val, y_pred_lgb, "LightGBM Validation")

    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------
    print("\nTrainiere CatBoost ...")
    cb_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.01,
        depth=8,
        loss_function="RMSE",
        early_stopping_rounds=50,
        verbose=False,
        random_seed=42,
    )

    cb_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val)
    )

    y_pred_cb = cb_model.predict(X_val)
    cb_metrics = evaluate(y_val, y_pred_cb, "CatBoost Validation")

    # --------------------------------------------------------
    # Modelle speichern
    # --------------------------------------------------------
    lgb_path = os.path.join(args.out_dir, f"lightgbm_{args.target}.txt")
    cb_path = os.path.join(args.out_dir, f"catboost_{args.target}.cbm")

    lgb_model.booster_.save_model(lgb_path)
    cb_model.save_model(cb_path)

    print(f"\nModelle gespeichert unter:\n  - {lgb_path}\n  - {cb_path}")

    # --------------------------------------------------------
    # Ergebnisse speichern
    # --------------------------------------------------------
    results = pd.DataFrame([lgb_metrics, cb_metrics])
    results.to_csv(os.path.join(args.out_dir, f"{args.target}_validation_metrics.csv"), index=False)

    # --------------------------------------------------------
    # Separate Auswertung: nur clean
    # --------------------------------------------------------
    if "snr" in df.columns:
        print("\n=== Separate Evaluation: Nur clean (SNR == 'clean') ===")
        clean_mask = df["snr"].astype(str).str.lower().eq("clean")
        if clean_mask.any():
            clean_df = df[clean_mask]
            X_clean = clean_df[feature_cols].astype(np.float32).values
            y_clean = clean_df[args.target].astype(np.float32).values

            # LightGBM clean
            y_pred_clean_lgb = lgb_model.predict(X_clean)
            metrics_clean_lgb = evaluate(y_clean, y_pred_clean_lgb, f"LightGBM (clean, {args.target})")

            # CatBoost clean
            y_pred_clean_cb = cb_model.predict(X_clean)
            metrics_clean_cb = evaluate(y_clean, y_pred_clean_cb, f"CatBoost (clean, {args.target})")

            # Speichern
            pd.DataFrame([metrics_clean_lgb, metrics_clean_cb]).to_csv(
                os.path.join(args.out_dir, f"{args.target}_validation_metrics_clean.csv"), index=False
            )
        else:
            print("Hinweis: Keine Zeilen mit snr == 'clean' gefunden.")


if __name__ == "__main__":
    main()