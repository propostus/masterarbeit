# scripts/evaluate_unseen_clean_augmented.py
# ------------------------------------------------------
# Evaluiert trainierte Modelle (LightGBM & CatBoost)
# auf einem Unseen-Datensatz (SigMOS + WavLM)
# Berechnet R², RMSE, MAE, CCC
# ------------------------------------------------------

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


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
# Evaluation einer einzelnen Modell-Datei
# ------------------------------------------------------
def evaluate_model(model_path, X, y, model_name, target):
    model = joblib.load(model_path)
    preds = model.predict(X)

    r2 = r2_score(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)
    ccc = concordance_correlation_coefficient(y, preds)

    print(f"{model_name:<20} ({target}): R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, CCC={ccc:.4f}")
    return {"model": model_name, "target": target, "r2": r2, "rmse": rmse, "mae": mae, "ccc": ccc}


# ------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LightGBM & CatBoost models on unseen data")
    parser.add_argument("--unseen_csv", required=True, help="Pfad zur Unseen-CSV (z. B. merged_sigmos_wavlm_unseen.csv)")
    parser.add_argument("--model_dir", required=True, help="Ordner mit gespeicherten Modellen (.pkl)")
    parser.add_argument("--out_csv", required=True, help="Pfad zur Ausgabedatei für Metriken (CSV)")
    args = parser.parse_args()

    print(f"=== Evaluation auf {args.unseen_csv} ===")

    df = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"Datensatz: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Feature-Spalten
    exclude_cols = [
        "filename", "reference", "hypothesis",
        "wer_tiny", "wer_base", "wer_small"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    results = []
    targets = ["wer_tiny", "wer_base", "wer_small"]

    for target in targets:
        y_true = df[target].astype(np.float32).values

        for model_name in ["LightGBM", "CatBoost"]:
            model_path = os.path.join(args.model_dir, f"{model_name}_{target}.pkl")
            if not os.path.exists(model_path):
                print(f"Modell nicht gefunden: {model_path}")
                continue

            X = df[feature_cols].astype(np.float32).values
            res = evaluate_model(model_path, X, y_true, model_name, target)
            results.append(res)

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)

    print("\n=== Fertig ===")
    print(results_df)
    print(f"\nMetriken gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()