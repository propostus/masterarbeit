# scripts/evaluate_tabular_unseen.py
# Evaluiert gespeicherte LightGBM- und CatBoost-Modelle auf einem unseen Dataset.

import os
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ------------------------------------------------------------
# Hilfsfunktion
# ------------------------------------------------------------
def evaluate(y_true, y_pred, label):
    """Berechnet R², RMSE, MAE."""
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{label}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    return {"model": label, "r2": r2, "rmse": rmse, "mae": mae}


# ------------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluiere LightGBM & CatBoost Modelle auf unseen Dataset")
    parser.add_argument("--dataset", required=True, help="Pfad zur unseen CSV")
    parser.add_argument("--target", required=True, help="Zielvariable (z. B. wer_tiny)")
    parser.add_argument("--model_dir", required=True, help="Ordner mit gespeicherten Modellen")
    parser.add_argument("--out_csv", required=True, help="Pfad zur Ergebnis-CSV")
    args = parser.parse_args()

    print(f"=== Evaluation auf {args.dataset} ===")

    # --------------------------------------------------------
    # Daten laden
    # --------------------------------------------------------
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Datensatz: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    exclude_cols = ["filename", "snr", "wer_tiny", "wer_base", "wer_small", "reference", "hypothesis"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    if args.target not in df.columns:
        raise ValueError(f"Zielspalte {args.target} fehlt in der CSV!")

    X = df[feature_cols].astype(np.float32).values
    y = df[args.target].astype(np.float32).values

    results = []

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------
    lgb_path = os.path.join(args.model_dir, f"lightgbm_{args.target}.txt")
    if os.path.exists(lgb_path):
        print(f"\n→ Lade LightGBM-Modell: {lgb_path}")
        lgb_model = lgb.Booster(model_file=lgb_path)
        y_pred_lgb = lgb_model.predict(X)
        metrics_lgb = evaluate(y, y_pred_lgb, "LightGBM Unseen")
        results.append(metrics_lgb)
    else:
        print("Kein LightGBM-Modell gefunden.")

    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------
    cb_path = os.path.join(args.model_dir, f"catboost_{args.target}.cbm")
    if os.path.exists(cb_path):
        print(f"\n→ Lade CatBoost-Modell: {cb_path}")
        cb_model = CatBoostRegressor()
        cb_model.load_model(cb_path)
        y_pred_cb = cb_model.predict(X)
        metrics_cb = evaluate(y, y_pred_cb, "CatBoost Unseen")
        results.append(metrics_cb)
    else:
        print("Kein CatBoost-Modell gefunden.")

    # --------------------------------------------------------
    # Ergebnisse speichern
    # --------------------------------------------------------
    if results:
        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        results_df.to_csv(args.out_csv, index=False)
        print(f"\nErgebnisse gespeichert unter: {args.out_csv}")
    else:
        print("\nKeine Modelle gefunden – keine Ergebnisse gespeichert.")


if __name__ == "__main__":
    main()