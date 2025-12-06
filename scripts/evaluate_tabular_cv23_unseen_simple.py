# scripts/evaluate_tabular_cv23_unseen_simple.py

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib


def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluiert CV23-Tabularmodelle (LightGBM + CatBoost, mit Handcrafted) auf dem Unseen-Datensatz."
    )
    parser.add_argument("--unseen_csv", required=True,
                        help="Pfad zu merged_sigmos_wavlm_unseen.csv")
    parser.add_argument("--handcrafted_unseen_csv", required=True,
                        help="Pfad zu handcrafted_audio_features_unseen_*.csv (gefiltert auf unseen)")
    parser.add_argument("--models_dir", required=True,
                        help="Verzeichnis mit LightGBM_with_handcrafted_*.pkl usw.")
    parser.add_argument("--out_csv", required=True,
                        help="Pfad zur Output-CSV mit den Metriken")
    args = parser.parse_args()

    # -----------------------------
    # 1) Daten laden und mergen
    # -----------------------------
    print(f"Lade Unseen-Basisdaten von: {args.unseen_csv}")
    df_unseen = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"Unseen-Basis: {df_unseen.shape[0]} Zeilen, {df_unseen.shape[1]} Spalten")

    print(f"Lade Handcrafted-Features von: {args.handcrafted_unseen_csv}")
    df_hand = pd.read_csv(args.handcrafted_unseen_csv)
    print(f"Handcrafted: {df_hand.shape[0]} Zeilen, {df_hand.shape[1]} Spalten")

    # Sanity-Check: nur Dateien, die wirklich im unseen-Set sind
    df_hand["filename"] = df_hand["filename"].astype(str)
    df_unseen["filename"] = df_unseen["filename"].astype(str)

    df = df_unseen.merge(df_hand, on="filename", how="left")
    print(f"Nach Merge (Unseen + Handcrafted): {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # -----------------------------
    # 2) Feature-Spalten wie im Training
    # -----------------------------
    exclude_cols = [
        "filename",
        "wer_tiny",
        "wer_base",
        "wer_small",
        "client_id",
        "age",
        "gender",
        "sentence",
    ]

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    print(f"Anzahl Feature-Spalten: {len(feature_cols)}")

    # X / y
    X = df[feature_cols].astype(np.float32).values
    targets = ["wer_tiny", "wer_base", "wer_small"]

    results = []

    # -----------------------------
    # 3) Modelle laden und evaluieren
    # -----------------------------
    for target in targets:
        y = df[target].astype(np.float32).values

        for model_prefix in ["LightGBM_with_handcrafted", "CatBoost_with_handcrafted"]:
            model_path = os.path.join(args.models_dir, f"{model_prefix}_{target}.pkl")
            if not os.path.exists(model_path):
                print(f"  Modell nicht gefunden: {model_path}")
                continue

            print(f"\n=== Evaluation: {model_prefix} ({target}) ===")
            model = joblib.load(model_path)

            preds = model.predict(X)

            r2 = r2_score(y, preds)
            rmse = mean_squared_error(y, preds, squared=False)
            mae = mean_absolute_error(y, preds)
            ccc = concordance_correlation_coefficient(y, preds)

            print(f"R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, CCC={ccc:.4f}")

            results.append({
                "model": model_prefix,
                "target": target,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "ccc": ccc,
            })

    if not results:
        print("Keine Ergebnisse – vermutlich keine Modelle gefunden.")
        return

    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    res_df.to_csv(args.out_csv, index=False)
    print(f"\nGesamtübersicht gespeichert unter: {args.out_csv}")
    print(res_df)


if __name__ == "__main__":
    main()