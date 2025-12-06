# scripts/evaluate_on_unseen_feature_noise.py
# ------------------------------------------------------
# Bewertet LightGBM & CatBoost Modelle (mit Feature-Noising)
# auf Unseen-Daten (merged_sigmos_wavlm_unseen.csv)
# Metriken: R², RMSE, MAE, CCC
# ------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


# ------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------

def concordance_correlation_coefficient(y_true, y_pred):
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)


def numeric_features(df, exclude_cols):
    return [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]


def evaluate_model(model_path, X, y, label):
    model = joblib.load(model_path)
    preds = model.predict(X)

    r2 = r2_score(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)
    ccc = concordance_correlation_coefficient(y, preds)

    return {
        "model": label,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "ccc": ccc
    }


# ------------------------------------------------------
# Hauptlogik
# ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Feature-Noise Modelle auf Unseen-Dataset")
    parser.add_argument("--unseen_csv", required=True, help="Pfad zur Unseen-CSV (z. B. merged_sigmos_wavlm_unseen.csv)")
    parser.add_argument("--models_dir", required=True, help="Ordner mit gespeicherten Modellen (.pkl)")
    parser.add_argument("--out_csv", required=True, help="Pfad für Ergebnis-CSV")
    args = parser.parse_args()

    # ------------------------------------------------------
    # Daten laden
    # ------------------------------------------------------
    df = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"Unseen geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    exclude_cols = ["filename", "group_id", "source", "reference", "hypothesis",
                    "wer_tiny", "wer_base", "wer_small"]
    feat_cols = numeric_features(df, exclude_cols)
    X = df[feat_cols].astype(np.float32).values

    targets = ["wer_tiny", "wer_base", "wer_small"]
    results = []

    # ------------------------------------------------------
    # Modelle evaluieren
    # ------------------------------------------------------
    for tgt in targets:
        if tgt not in df.columns:
            print(f"WARNUNG: Spalte {tgt} fehlt im Unseen-Datensatz – übersprungen.")
            continue

        y = df[tgt].astype(np.float32).values

        for model_name in ["LightGBM", "CatBoost"]:
            model_path = os.path.join(args.models_dir, f"{model_name}_{tgt}_featnoise.pkl")
            if not os.path.exists(model_path):
                print(f"Modell nicht gefunden: {model_path}")
                continue

            res = evaluate_model(model_path, X, y, f"{model_name} ({tgt}, feat-noise)")
            results.append(res)

    # Ergebnisse speichern
    res_df = pd.DataFrame(results)
    res_df.to_csv(args.out_csv, index=False)

    print("\n=== Unseen Evaluation (Feature-Noising) ===")
    print(res_df)
    print(f"\nMetriken gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()