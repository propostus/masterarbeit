# scripts/evaluate_on_unseen_with_pca.py
# ------------------------------------------------------
# Lädt *_pca.pkl Modelle (LightGBM / CatBoost),
# wendet gespeicherte PCA auf Unseen-Features an
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

def numeric_feature_cols(df, exclude_cols):
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
    ap = argparse.ArgumentParser(description="Evaluate PCA models on Unseen dataset")
    ap.add_argument("--unseen_csv", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--target", required=True, choices=["wer_tiny", "wer_base", "wer_small"])
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"Unseen geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    exclude = ["filename", "group_id", "source", "reference", "hypothesis",
               "wer_tiny", "wer_base", "wer_small"]
    feats = numeric_feature_cols(df, exclude)
    df = df.dropna(subset=[args.target])
    X_full = df[feats].astype(np.float32).values
    y = df[args.target].astype(np.float32).values

    # PCA laden und anwenden
    pca_path = os.path.join(args.models_dir, "pca_transformer.pkl")
    pca = joblib.load(pca_path)
    X = pca.transform(X_full)
    print(f"PCA angewendet: {X.shape[1]} Dimensionen")

    results = []
    lgb_path = os.path.join(args.models_dir, f"LightGBM_{args.target}_pca.pkl")
    cat_path = os.path.join(args.models_dir, f"CatBoost_{args.target}_pca.pkl")

    if os.path.exists(lgb_path):
        eval_one(lgb_path, X, y, f"LightGBM ({args.target}, PCA)", results)
    if os.path.exists(cat_path):
        eval_one(cat_path, X, y, f"CatBoost ({args.target}, PCA)", results)

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.out_csv, index=False)
    print("\n=== Unseen Evaluation (PCA) ===")
    print(res_df)
    print(f"\nErgebnisse gespeichert unter: {args.out_csv}")

if __name__ == "__main__":
    main()