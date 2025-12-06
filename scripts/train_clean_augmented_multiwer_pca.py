# scripts/train_clean_augmented_multiwer_pca.py
# ------------------------------------------------------
# Trainiert LightGBM & CatBoost auf Clean + Augmented Dataset
# mit GroupSplit, PCA-Reduktion der Features (zielabhängig, z. B. 128D)
# und Evaluation (R2, RMSE, MAE, CCC)
# ------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib


# ------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------

def concordance_correlation_coefficient(y_true, y_pred):
    """Berechnet den Concordance Correlation Coefficient (CCC)."""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)


def numeric_feature_cols(df, exclude_cols):
    """Filtert alle numerischen Feature-Spalten (außer Ziel- und Meta-Spalten)."""
    return [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]


def train_and_eval_model(X_tr, X_te, y_tr, y_te, model, model_name, target, out_dir):
    """Trainiert Modell, evaluiert und speichert es."""
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    r2 = r2_score(y_te, preds)
    rmse = mean_squared_error(y_te, preds, squared=False)
    mae = mean_absolute_error(y_te, preds)
    ccc = concordance_correlation_coefficient(y_te, preds)

    joblib.dump(model, os.path.join(out_dir, f"{model_name}_{target}_pca.pkl"))

    return {
        "model": model_name,
        "target": target,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "ccc": ccc
    }


# ------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM & CatBoost on Clean+Augmented dataset (PCA pro Target)")
    parser.add_argument("--dataset", required=True, help="Pfad zur kombinierten Clean+Augmented CSV")
    parser.add_argument("--out_dir", required=True, help="Verzeichnis zum Speichern der Modelle und Metriken")
    parser.add_argument("--pca_dim", type=int, default=128, help="Zieldimension der PCA")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"=== Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten ===")

    # Spalten bestimmen
    exclude = [
        "filename", "group_id", "source",
        "reference", "hypothesis",
        "wer_tiny", "wer_base", "wer_small"
    ]
    features = numeric_feature_cols(df, exclude)

    # Gruppierter Split (kein Leakage)
    if "group_id" not in df.columns:
        df["group_id"] = df["filename"].str.replace(r"\.(wav|mp3|flac)$", "", regex=True)
        df["group_id"] = df["group_id"].str.replace(r"_aug$", "", regex=True)

    groups = df["group_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(df, groups=groups))
    df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]

    print(f"Train: {len(df_tr)}  |  Test: {len(df_te)}")
    print(f"Unique groups (Train/Test): {df_tr['group_id'].nunique()} / {df_te['group_id'].nunique()}")

    results = []
    targets = ["wer_tiny", "wer_base", "wer_small"]

    # ======================================================
    # Zielvariablen-spezifisches Training (mit eigener PCA)
    # ======================================================
    for tgt in targets:
        print(f"\n=== Training für Zielvariable: {tgt} (PCA={args.pca_dim}) ===")

        df_tr_t = df_tr.dropna(subset=[tgt])
        df_te_t = df_te.dropna(subset=[tgt])

        X_tr_full = df_tr_t[features].astype(np.float32).values
        X_te_full = df_te_t[features].astype(np.float32).values
        y_tr = df_tr_t[tgt].astype(np.float32).values
        y_te = df_te_t[tgt].astype(np.float32).values

        # PCA fit + transform
        pca = PCA(n_components=args.pca_dim, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_full)
        X_te_pca = pca.transform(X_te_full)

        pca_path = os.path.join(args.out_dir, f"pca_{tgt}.pkl")
        joblib.dump(pca, pca_path)
        print(f"PCA gespeichert unter: {pca_path}")

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
            random_seed=42,
        )

        # Train & Eval
        results.append(train_and_eval_model(X_tr_pca, X_te_pca, y_tr, y_te, lgbm, "LightGBM", tgt, args.out_dir))
        results.append(train_and_eval_model(X_tr_pca, X_te_pca, y_tr, y_te, cat, "CatBoost", tgt, args.out_dir))

    # Ergebnisse speichern
    res_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "train_metrics_clean_augmented_pca.csv")
    res_df.to_csv(out_csv, index=False)
    print("\n=== Fertig (PCA Training) ===")
    print(res_df)
    print(f"\nMetriken gespeichert unter: {out_csv}")


if __name__ == "__main__":
    main()