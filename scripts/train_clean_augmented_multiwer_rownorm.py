# scripts/train_clean_augmented_multiwer_rownorm.py
# ------------------------------------------------------
# Trainiert LightGBM & CatBoost auf Clean + Augmented Dataset
# mit GroupSplit, ROW-WISE Normalisierung (pro Sample),
# mehreren Targets (wer_tiny/base/small) und Metriken (R2, RMSE, MAE, CCC).
# Speichert Modelle als *_rownorm.pkl.
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

# ---------- Metrics ----------
def ccc(y_true, y_pred):
    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    return (2 * cov) / (var_t + var_p + (mu_t - mu_p) ** 2 + 1e-12)

# ---------- Row-wise Z-Score ----------
def row_zscore(X: np.ndarray) -> np.ndarray:
    # z = (x - mean_row) / std_row ; stabil mit epsilon
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    return (X - m) / (s + 1e-8)

def numeric_feature_cols(df: pd.DataFrame, exclude_cols):
    return [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

def train_and_eval_model(X_tr, X_te, y_tr, y_te, model, model_name, target, out_dir):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    res = {
        "model": model_name,
        "target": target,
        "r2": r2_score(y_te, pred),
        "rmse": mean_squared_error(y_te, pred, squared=False),
        "mae": mean_absolute_error(y_te, pred),
        "ccc": ccc(y_te, pred),
    }
    joblib.dump(model, os.path.join(out_dir, f"{model_name}_{target}_rownorm.pkl"))
    return res

def main():
    ap = argparse.ArgumentParser(description="Train LGBM & CatBoost with row-wise normalization on Clean+Augmented")
    ap.add_argument("--dataset", required=True, help="CSV: clean+augmented zusammengeführt")
    ap.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"=== Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten ===")

    # Spalten definieren
    exclude = ["filename", "group_id", "source", "reference", "hypothesis",
               "wer_tiny", "wer_base", "wer_small"]
    feats = numeric_feature_cols(df, exclude)

    # GroupSplit (clean+aug zusammenhalten)
    if "group_id" not in df.columns:
        df["group_id"] = df["filename"].str.replace(r"\.(wav|mp3|flac)$", "", regex=True)
        # optional: _aug entfernen, falls vorhanden
        df["group_id"] = df["group_id"].str.replace(r"_aug$", "", regex=True)

    groups = df["group_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(df, groups=groups))
    df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
    print(f"Train: {len(df_tr)}  |  Test: {len(df_te)}")
    print(f"Unique groups (Train/Test): {df_tr['group_id'].nunique()} / {df_te['group_id'].nunique()}")

    targets = ["wer_tiny", "wer_base", "wer_small"]
    results = []

    for tgt in targets:
        print(f"\n=== Training für Zielvariable: {tgt} (row-wise normalisiert) ===")
        # Drop NaNs im Ziel
        df_tr_t = df_tr.dropna(subset=[tgt])
        df_te_t = df_te.dropna(subset=[tgt])

        X_tr = df_tr_t[feats].astype(np.float32).values
        y_tr = df_tr_t[tgt].astype(np.float32).values
        X_te = df_te_t[feats].astype(np.float32).values
        y_te = df_te_t[tgt].astype(np.float32).values

        # Row-wise Normalisierung (nur Features)
        X_tr = row_zscore(X_tr)
        X_te = row_zscore(X_te)

        lgbm = LGBMRegressor(
            n_estimators=400, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9,
            max_depth=-1, random_state=42
        )
        cat = CatBoostRegressor(
            iterations=400, learning_rate=0.03,
            depth=8, loss_function="RMSE",
            random_seed=42, verbose=0
        )

        results.append(train_and_eval_model(X_tr, X_te, y_tr, y_te, lgbm, "LightGBM", tgt, args.out_dir))
        results.append(train_and_eval_model(X_tr, X_te, y_tr, y_te, cat,  "CatBoost",  tgt, args.out_dir))

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "train_metrics_clean_augmented_rownorm.csv")
    res_df.to_csv(out_csv, index=False)
    print("\n=== Fertig (Row-Norm) ===")
    print(res_df)
    print(f"\nMetriken gespeichert unter: {out_csv}")

if __name__ == "__main__":
    main()