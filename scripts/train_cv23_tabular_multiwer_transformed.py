# scripts/train_cv23_tabular_multiwer_transformed.py
# ------------------------------------------------------
# Trainiert LightGBM & CatBoost auf CV23-balanced
# (SigMOS + WavLM, Targets: wer_tiny / wer_base / wer_small)
# mit optionaler Zieltransformation:
#   --target_transform none | log | sqrt
# Metriken werden IMMER auf der Original-WER-Skala berechnet.
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


def concordance_correlation_coefficient(y_true, y_pred):
    """Berechnet den Concordance Correlation Coefficient (CCC)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc


def apply_target_transform(y, transform, eps):
    """Transformiert die Zielvariable für das Training."""
    if transform == "none":
        return y
    y = np.asarray(y, dtype=np.float32)
    if transform == "log":
        return np.log(y + eps)
    elif transform == "sqrt":
        return np.sqrt(y)
    else:
        raise ValueError(f"Unbekannte target_transform: {transform}")


def invert_target_transform(y_pred, transform, eps):
    """Invertiert die Zieltransformation für die Metrikberechnung."""
    if transform == "none":
        return y_pred
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if transform == "log":
        return np.exp(y_pred) - eps
    elif transform == "sqrt":
        return np.square(y_pred)
    else:
        raise ValueError(f"Unbekannte target_transform: {transform}")


def train_and_eval_model(
    X_train,
    X_test,
    y_train_raw,
    y_test_raw,
    model,
    model_name,
    target_name,
    out_dir,
    target_transform="none",
    eps=1e-4,
):
    """
    Trainiert ein Modell auf transformierter Zielvariable und
    berechnet Metriken auf der Original-WER-Skala.
    """
    # Ziel für Training transformieren
    y_train = apply_target_transform(y_train_raw, target_transform, eps)

    # Training
    model.fit(X_train, y_train)

    # Vorhersage im Transform-Space
    y_pred_transformed = model.predict(X_test)

    # Zurück auf Originalskala
    y_pred = invert_target_transform(y_pred_transformed, target_transform, eps)

    # Metriken auf Original-WER-Skala
    r2 = r2_score(y_test_raw, y_pred)
    rmse = mean_squared_error(y_test_raw, y_pred, squared=False)
    mae = mean_absolute_error(y_test_raw, y_pred)
    ccc = concordance_correlation_coefficient(y_test_raw, y_pred)

    # Modell speichern
    model_path = os.path.join(out_dir, f"{model_name}_{target_name}.pkl")
    joblib.dump(model, model_path)

    return {
        "model": model_name,
        "target": target_name,
        "target_transform": target_transform,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "ccc": ccc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM & CatBoost auf CV23-balanced (multi-WER) mit Zieltransformation."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Pfad zur CSV, z. B. results/datasets/merged_sigmos_wavlm_cv23_balanced_multiwer.csv",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Verzeichnis zum Speichern der Modelle und Metriken",
    )
    parser.add_argument(
        "--target_transform",
        choices=["none", "log", "sqrt"],
        default="none",
        help="Transformation der Zielvariable vor dem Training",
    )
    parser.add_argument(
        "--log_eps",
        type=float,
        default=1e-4,
        help="Epsilon für log-Transformation: log(wer + eps)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== Training auf {args.dataset} ===")
    print(f"Zieltransformation: {args.target_transform}")

    # Daten laden
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Features bestimmen: nur numerische Spalten, ohne Leckage / Meta-Info
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
        c
        for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    print(f"Anzahl Feature-Spalten: {len(feature_cols)}")

    # Gruppen-Split nach Sprecher:innen (client_id)
    if "client_id" not in df.columns:
        raise RuntimeError("Spalte 'client_id' wird für den Gruppensplit benötigt.")

    groups = df["client_id"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")
    print(
        f"Unique Sprecher:innen (Train/Test): "
        f"{train_df['client_id'].nunique()} / {test_df['client_id'].nunique()}"
    )

    # X / y vorbereiten
    X_train = train_df[feature_cols].astype(np.float32).values
    X_test = test_df[feature_cols].astype(np.float32).values

    targets = ["wer_tiny", "wer_base", "wer_small"]
    results = []

    for target in targets:
        print(f"\n=== Training für Zielvariable: {target} ===")

        y_train_raw = train_df[target].astype(np.float32).values
        y_test_raw = test_df[target].astype(np.float32).values

        # LightGBM
        lgbm = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )

        # CatBoost
        cat = CatBoostRegressor(
            iterations=400,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            verbose=0,
            random_seed=42,
        )

        res_lgb = train_and_eval_model(
            X_train,
            X_test,
            y_train_raw,
            y_test_raw,
            lgbm,
            "LightGBM",
            target,
            args.out_dir,
            target_transform=args.target_transform,
            eps=args.log_eps,
        )
        res_cat = train_and_eval_model(
            X_train,
            X_test,
            y_train_raw,
            y_test_raw,
            cat,
            "CatBoost",
            target,
            args.out_dir,
            target_transform=args.target_transform,
            eps=args.log_eps,
        )

        results.extend([res_lgb, res_cat])

    results_df = pd.DataFrame(results)
    out_csv = os.path.join(
        args.out_dir, f"train_metrics_cv23_balanced_{args.target_transform}.csv"
    )
    results_df.to_csv(out_csv, index=False)

    print("\n=== Training abgeschlossen (CV23-balanced) ===")
    print(results_df)
    print(f"\nMetriken gespeichert unter: {out_csv}")


if __name__ == "__main__":
    main()