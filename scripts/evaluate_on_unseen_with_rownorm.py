# scripts/evaluate_on_unseen_with_rownorm.py
# ------------------------------------------------------
# Lädt *_rownorm.pkl Modelle (LightGBM / CatBoost),
# wendet identische ROW-WISE Normalisierung auf Unseen-CSV an
# und berechnet R2, RMSE, MAE, CCC pro Zielvariable.
# ------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def ccc(y_true, y_pred):
    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    return (2 * cov) / (var_t + var_p + (mu_t - mu_p) ** 2 + 1e-12)

def row_zscore(X: np.ndarray) -> np.ndarray:
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    return (X - m) / (s + 1e-8)

def numeric_feature_cols(df: pd.DataFrame, exclude_cols):
    return [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

def eval_one(model_path, X, y, label, results):
    model = joblib.load(model_path)
    pred = model.predict(X)
    results.append({
        "model": label,
        "r2": r2_score(y, pred),
        "rmse": mean_squared_error(y, pred, squared=False),
        "mae": mean_absolute_error(y, pred),
        "ccc": ccc(y, pred),
    })

def main():
    ap = argparse.ArgumentParser(description="Evaluate *_rownorm.pkl models on unseen CSV with row-wise normalization")
    ap.add_argument("--unseen_csv", required=True, help="CSV mit Unseen-Features (sigmos + wavlm)")
    ap.add_argument("--out_csv", required=True, help="Wohin die Metriken geschrieben werden")
    ap.add_argument("--models_dir", required=True, help="Ordner mit gespeicherten *_rownorm.pkl Modellen")
    ap.add_argument("--target", required=True, choices=["wer_tiny", "wer_base", "wer_small"])
    args = ap.parse_args()

    df = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"Unseen geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    exclude = ["filename", "group_id", "source", "reference", "hypothesis",
               "wer_tiny", "wer_base", "wer_small"]
    feats = numeric_feature_cols(df, exclude)

    # Nur Zeilen mit gültigem Zielwert (falls vorhanden)
    df = df.dropna(subset=[args.target])

    X = df[feats].astype(np.float32).values
    y = df[args.target].astype(np.float32).values

    # Row-wise Normalisierung anwenden (wie im Training)
    X = row_zscore(X)

    results = []
    lgb_path = os.path.join(args.models_dir, f"LightGBM_{args.target}_rownorm.pkl")
    cat_path = os.path.join(args.models_dir, f"CatBoost_{args.target}_rownorm.pkl")

    if os.path.exists(lgb_path):
        eval_one(lgb_path, X, y, f"LightGBM ({args.target}, rownorm)", results)
    else:
        print(f"Warnung: {lgb_path} nicht gefunden.")

    if os.path.exists(cat_path):
        eval_one(cat_path, X, y, f"CatBoost ({args.target}, rownorm)", results)
    else:
        print(f"Warnung: {cat_path} nicht gefunden.")

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.out_csv, index=False)
    print("\n=== Unseen Evaluation (Row-Norm) ===")
    print(res_df)
    print(f"\nErgebnisse gespeichert unter: {args.out_csv}")

if __name__ == "__main__":
    main()