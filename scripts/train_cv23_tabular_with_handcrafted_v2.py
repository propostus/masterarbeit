import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
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
        description="Trainiert LightGBM & CatBoost auf CV23-balanced + Handcrafted (v2, mit gespeicherten Feature-Cols)"
    )
    parser.add_argument("--dataset", required=True,
                        help="merged_sigmos_wavlm_cv23_balanced_multiwer.csv")
    parser.add_argument("--handcrafted_csv", required=True,
                        help="handcrafted_audio_features_cv23_balanced.csv")
    parser.add_argument("--out_dir", required=True,
                        help="Ausgabeverzeichnis für Modelle + Featureliste")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Daten laden & mergen ---
    print(f"Lade Basisdaten von: {args.dataset}")
    df_base = pd.read_csv(args.dataset, low_memory=False)
    print(f"Basis: {df_base.shape[0]} Zeilen, {df_base.shape[1]} Spalten")

    print(f"Lade Handcrafted-Features von: {args.handcrafted_csv}")
    df_hand = pd.read_csv(args.handcrafted_csv)
    print(f"Handcrafted: {df_hand.shape[0]} Zeilen, {df_hand.shape[1]} Spalten")

    df_base["filename"] = df_base["filename"].astype(str)
    df_hand["filename"] = df_hand["filename"].astype(str)

    df = df_base.merge(df_hand, on="filename", how="left")
    print(f"Nach Merge: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Optional: Zeilen mit NaN-Zielen droppen
    df = df.dropna(subset=["wer_tiny", "wer_base", "wer_small"])
    print(f"Nach Drop NaN-Targets: {df.shape[0]} Zeilen")

    # --- Feature-Spalten bestimmen UND speichern ---
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

    # Featureliste speichern – damit Evaluation 100% reproduzierbar ist
    feature_path = os.path.join(args.out_dir, "feature_cols.txt")
    with open(feature_path, "w") as f:
        for c in feature_cols:
            f.write(c + "\n")
    print(f"Featureliste gespeichert unter: {feature_path}")

    # --- GroupSplit nach client_id ---
    if "client_id" in df.columns:
        groups = df["client_id"].astype(str)
    else:
        groups = df["filename"].astype(str)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_idx, test_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Unique Speaker (Train/Test): "
          f"{train_df['client_id'].nunique() if 'client_id' in df.columns else 'n/a'} / "
          f"{test_df['client_id'].nunique() if 'client_id' in df.columns else 'n/a'}")

    X_train = train_df[feature_cols].astype(np.float32).values
    X_test = test_df[feature_cols].astype(np.float32).values

    results = []
    targets = ["wer_tiny", "wer_base", "wer_small"]

    for target in targets:
        print(f"\n=== Training für Zielvariable: {target} ===")
        y_train = train_df[target].astype(np.float32).values
        y_test = test_df[target].astype(np.float32).values

        lgbm = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        cat = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )

        # LightGBM
        lgbm.fit(X_train, y_train)
        preds_lgb = lgbm.predict(X_test)
        r2 = r2_score(y_test, preds_lgb)
        rmse = mean_squared_error(y_test, preds_lgb, squared=False)
        mae = mean_absolute_error(y_test, preds_lgb)
        ccc = concordance_correlation_coefficient(y_test, preds_lgb)

        joblib.dump(
            lgbm,
            os.path.join(args.out_dir, f"LGBM_v2_{target}.pkl")
        )

        results.append({
            "model": "LGBM_v2",
            "target": target,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "ccc": ccc,
        })

        # CatBoost
        cat.fit(X_train, y_train)
        preds_cat = cat.predict(X_test)
        r2 = r2_score(y_test, preds_cat)
        rmse = mean_squared_error(y_test, preds_cat, squared=False)
        mae = mean_absolute_error(y_test, preds_cat)
        ccc = concordance_correlation_coefficient(y_test, preds_cat)

        joblib.dump(
            cat,
            os.path.join(args.out_dir, f"CatBoost_v2_{target}.pkl")
        )

        results.append({
            "model": "CatBoost_v2",
            "target": target,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "ccc": ccc,
        })

    res_df = pd.DataFrame(results)
    metrics_path = os.path.join(args.out_dir, "train_metrics_tabular_v2.csv")
    res_df.to_csv(metrics_path, index=False)
    print("\n=== Training abgeschlossen (v2) ===")
    print(res_df)
    print(f"\nMetriken gespeichert unter: {metrics_path}")


if __name__ == "__main__":
    main()