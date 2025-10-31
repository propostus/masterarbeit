# scripts/train_and_compare_normalized.py
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def train_and_evaluate(X, y, model, model_name, is_lgbm=False):
    """Trainiert Modell und gibt R², RMSE, MAE zurück"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBM / CatBoost anpassen
    if is_lgbm:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mae = mean_absolute_error(y_val, y_pred)
    return {"model": model_name, "r2": r2, "rmse": rmse, "mae": mae}


def load_features(path, target):
    """Lädt Dataset, wählt numerische Features + Zielvariable"""
    df = pd.read_csv(path, low_memory=False)
    exclude_cols = ["filename", "snr", "wer_tiny", "wer_base", "wer_small", "reference", "hypothesis"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
    X = df[feature_cols].astype(np.float32)
    y = df[target].astype(np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Vergleiche Training mit Original- vs. Normalized-Dataset")
    parser.add_argument("--dataset_original", required=True, help="Pfad zur Original-CSV (z. B. embeddings_sigmos_wavlm_clean_and_noisy_tiny.csv)")
    parser.add_argument("--dataset_normalized", required=True, help="Pfad zur normalisierten CSV (z. B. embeddings_sigmos_wavlm_clean_and_noisy_tiny_normalized.csv)")
    parser.add_argument("--target", default="wer_tiny", help="Zielvariable (z. B. wer_tiny)")
    parser.add_argument("--out_csv", required=True, help="Pfad für Ergebnisvergleich")
    args = parser.parse_args()

    print(f"=== Vergleich Original vs. Normalized (Target: {args.target}) ===")

    results = []

    for version_name, path in [("Original", args.dataset_original), ("Normalized", args.dataset_normalized)]:
        print(f"\n--- {version_name} Dataset ---")
        X, y = load_features(path, args.target)

        # LightGBM
        lgb = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        res_lgb = train_and_evaluate(X, y, lgb, f"LGBM ({version_name})", is_lgbm=True)
        results.append(res_lgb)

        # CatBoost
        cat = CatBoostRegressor(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            loss_function="RMSE",
            verbose=False,
            random_state=42
        )
        res_cat = train_and_evaluate(X, y, cat, f"CatBoost ({version_name})")
        results.append(res_cat)

    # Ergebnisse als Tabelle
    df_results = pd.DataFrame(results)
    print("\n=== Vergleich: Original vs. Normalized ===")
    print(df_results)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_results.to_csv(args.out_csv, index=False)
    print(f"\nErgebnisse gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()