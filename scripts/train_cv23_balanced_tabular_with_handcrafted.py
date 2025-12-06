# scripts/train_cv23_balanced_tabular_with_handcrafted.py
# ------------------------------------------------------
# Trainiert LightGBM & CatBoost auf CV23-balanced (SigMOS + WavLM)
# + optionale Handcrafted-Audiofeatures.
# - Gruppierter Split nach client_id (kein Sprecher-Leakage)
# - Mehrere Targets: wer_tiny, wer_base, wer_small
# - Metriken: R2, RMSE, MAE, CCC
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
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)
    return float(ccc)


def train_and_eval_model(X_train, X_test, y_train, y_test, model, model_name, target_name, out_dir):
    """Trainiert ein Modell und berechnet Metriken auf dem Test-Set."""
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    ccc = concordance_correlation_coefficient(y_test, preds)

    # Modell speichern
    model_path = os.path.join(out_dir, f"{model_name}_{target_name}.pkl")
    joblib.dump(model, model_path)

    return {
        "model": model_name,
        "target": target_name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "ccc": ccc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Trainiert LightGBM & CatBoost auf CV23-balanced (SigMOS+WavLM) mit optionalen Handcrafted-Audiofeatures."
    )
    parser.add_argument("--dataset", required=True, help="Pfad zur CV23-balanced Multi-WER CSV (SigMOS + WavLM)")
    parser.add_argument("--extra_features_csv", required=True, help="Pfad zu den Handcrafted-Audiofeatures (CSV, mit Spalte 'filename')")
    parser.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis für Modelle und Metriken")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test-Anteil für GroupShuffleSplit (Standard: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random Seed für GroupShuffleSplit (Standard: 42)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --------------------------------------------------
    # 1) Haupt-Datensatz laden
    # --------------------------------------------------
    print(f"Lade Hauptdatensatz: {args.dataset}")
    df_main = pd.read_csv(args.dataset, low_memory=False)
    print(f"Hauptdatensatz: {df_main.shape[0]} Zeilen, {df_main.shape[1]} Spalten")

    if "filename" not in df_main.columns:
        raise RuntimeError("Im Hauptdatensatz wird eine Spalte 'filename' erwartet.")

    # --------------------------------------------------
    # 2) Handcrafted-Features laden und mergen
    # --------------------------------------------------
    print(f"Lade Handcrafted-Features von: {args.extra_features_csv}")
    df_extra = pd.read_csv(args.extra_features_csv, low_memory=False)
    if "filename" not in df_extra.columns:
        raise RuntimeError("In der Extra-Feature-CSV wird eine Spalte 'filename' erwartet.")

    before_merge = df_main.shape[0]
    df = df_main.merge(df_extra, on="filename", how="inner")
    after_merge = df.shape[0]

    print(f"Nach Merge mit Handcrafted-Features: {after_merge} Zeilen (vorher {before_merge})")
    if after_merge < before_merge:
        print("Hinweis: Es wurden Zeilen verworfen, weil keine Handcrafted-Features für manche Dateien vorhanden waren.")

    # --------------------------------------------------
    # 3) Targets und Features vorbereiten
    # --------------------------------------------------
    targets = ["wer_tiny", "wer_base", "wer_small"]
    for t in targets:
        if t not in df.columns:
            raise RuntimeError(f"Target-Spalte '{t}' fehlt im Datensatz.")

    # NaNs in Targets entfernen
    df = df.dropna(subset=targets)

    # Gruppierung nach Sprecher: client_id
    if "client_id" not in df.columns:
        raise RuntimeError("Spalte 'client_id' wird zur Gruppierung benötigt.")
    groups = df["client_id"]

    # Nicht als Features zu verwendende Spalten
    exclude_cols = set(
        [
            "filename",
            "client_id",
            "age",
            "gender",
            "sentence",
            "wer_tiny",
            "wer_base",
            "wer_small",
        ]
    )

    # Numerische Feature-Spalten bestimmen
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    print(f"Anzahl Feature-Spalten (inkl. Handcrafted): {len(feature_cols)}")

    # Features extrahieren und NaNs in Features mit Spaltenmittelwert füllen
    X_all = df[feature_cols].astype(np.float32)
    X_all = X_all.fillna(X_all.mean())

    # --------------------------------------------------
    # 4) GroupShuffleSplit (Train/Test)
    # --------------------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.random_state
    )
    train_idx, test_idx = next(splitter.split(X_all, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)} Zeilen, Test: {len(test_df)} Zeilen")
    print(f"Einzigartige Sprecher (Train/Test): {train_df['client_id'].nunique()} / {test_df['client_id'].nunique()}")

    X_train = train_df[feature_cols].astype(np.float32)
    X_train = X_train.fillna(X_train.mean())
    X_test = test_df[feature_cols].astype(np.float32)
    X_test = X_test.fillna(X_test.mean())

    # --------------------------------------------------
    # 5) Modelle definieren
    # --------------------------------------------------
    results = []

    for target in targets:
        print(f"\n=== Training für Zielvariable: {target} ===")

        y_train = train_df[target].astype(np.float32).values
        y_test = test_df[target].astype(np.float32).values

        # LightGBM
        lgbm = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=args.random_state,
        )

        # CatBoost
        cat = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            verbose=0,
            random_seed=args.random_state,
        )

        res_lgb = train_and_eval_model(
            X_train.values, X_test.values, y_train, y_test,
            lgbm, "LightGBM_with_handcrafted", target, args.out_dir
        )
        res_cat = train_and_eval_model(
            X_train.values, X_test.values, y_train, y_test,
            cat, "CatBoost_with_handcrafted", target, args.out_dir
        )

        results.extend([res_lgb, res_cat])

    # --------------------------------------------------
    # 6) Metriken speichern
    # --------------------------------------------------
    results_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "train_metrics_cv23_balanced_with_handcrafted.csv")
    results_df.to_csv(out_csv, index=False)

    print("\n=== Training abgeschlossen (CV23-balanced + Handcrafted) ===")
    print(results_df)
    print(f"\nMetriken gespeichert unter: {out_csv}")
    print(f"Modelle gespeichert in: {args.out_dir}")


if __name__ == "__main__":
    main()