# scripts/train_clean_augmented_multiwer_feature_noise.py
# ------------------------------------------------------
# Trainiert LightGBM & CatBoost auf Clean+Augmented Dataset
# mit GroupSplit, Feature-Noising & Feature-Masking
# zur Erhöhung der Robustheit auf Unseen-Daten.
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

# ------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------

def concordance_correlation_coefficient(y_true, y_pred):
    """Berechnet den Concordance Correlation Coefficient (CCC)."""
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)


def add_feature_noise(X, noise_std=0.01, mask_prob=0.05, random_state=42):
    """Fügt Rauschen und zufällige Maskierung in Feature-Matrix hinzu."""
    rng = np.random.default_rng(random_state)

    # Gaußsches Rauschen
    noise = rng.normal(0, noise_std, X.shape).astype(np.float32)
    X_noised = X + noise

    # Masking: zufällige Features auf 0 setzen
    mask = rng.random(X.shape) < mask_prob
    X_noised[mask] = 0.0

    return X_noised


def train_and_eval_model(X_train, X_test, y_train, y_test, model, model_name, target_name, out_dir):
    """Trainiert, evaluiert und speichert Modell."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    ccc = concordance_correlation_coefficient(y_test, preds)

    joblib.dump(model, os.path.join(out_dir, f"{model_name}_{target_name}_featnoise.pkl"))

    return {"model": model_name, "target": target_name, "r2": r2, "rmse": rmse, "mae": mae, "ccc": ccc}


# ------------------------------------------------------
# Hauptlogik
# ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM & CatBoost mit Feature-Noising + Masking")
    parser.add_argument("--dataset", required=True, help="Pfad zur kombinierten Clean+Augmented CSV")
    parser.add_argument("--out_dir", required=True, help="Verzeichnis für Modelle & Metriken")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Standardabweichung des Feature-Rauschens")
    parser.add_argument("--mask_prob", type=float, default=0.05, help="Wahrscheinlichkeit, ein Feature auf 0 zu setzen")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"=== Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten ===")

    # Feature-Spalten
    exclude_cols = ["filename", "group_id", "source", "wer_tiny", "wer_base", "wer_small", "reference", "hypothesis"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    # Gruppierter Split (kein Leakage zwischen clean/aug)
    if "group_id" not in df.columns:
        df["group_id"] = df["filename"].str.replace(r"\.(wav|mp3|flac)$", "", regex=True)
        df["group_id"] = df["group_id"].str.replace(r"_aug$", "", regex=True)

    groups = df["group_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")
    print(f"Unique groups (Train/Test): {train_df['group_id'].nunique()} / {test_df['group_id'].nunique()}")

    results = []
    targets = ["wer_tiny", "wer_base", "wer_small"]

    # ------------------------------------------------------
    # Training pro Zielvariable mit Feature Noise & Masking
    # ------------------------------------------------------
    for target in targets:
        print(f"\n=== Training für Zielvariable: {target} (Noise={args.noise_std}, Mask={args.mask_prob}) ===")

        df_tr = train_df.dropna(subset=[target])
        df_te = test_df.dropna(subset=[target])

        X_train = df_tr[feature_cols].astype(np.float32).values
        y_train = df_tr[target].astype(np.float32).values
        X_test = df_te[feature_cols].astype(np.float32).values
        y_test = df_te[target].astype(np.float32).values

        # Feature Noise / Masking anwenden
        X_train_noised = add_feature_noise(X_train, args.noise_std, args.mask_prob, random_state=42)

        # Modelle definieren
        lgbm = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            max_depth=-1,
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

        res_lgb = train_and_eval_model(X_train_noised, X_test, y_train, y_test, lgbm, "LightGBM", target, args.out_dir)
        res_cat = train_and_eval_model(X_train_noised, X_test, y_train, y_test, cat, "CatBoost", target, args.out_dir)

        results.extend([res_lgb, res_cat])

    results_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "train_metrics_feature_noise.csv")
    results_df.to_csv(out_csv, index=False)
    print("\n=== Fertig! ===")
    print(results_df)
    print(f"\nMetriken gespeichert unter: {out_csv}")


if __name__ == "__main__":
    main()