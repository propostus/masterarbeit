# scripts/eval_cv23_tabular_multiwer_v3_on_unseen.py
# ------------------------------------------------------
# Lädt die v3-Tabular-Modelle (nur SigMOS+WavLM),
# wendet sie auf den Unseen-Datensatz an und berechnet:
#   R², RMSE, MAE, CCC
#
# Erwartet:
#   - unseen_csv: merged_sigmos_wavlm_unseen.csv
#   - models_dir: Ordner mit LGBM_v3_<target>.pkl, CatBoost_v3_<target>.pkl,
#                 feature_cols_v3.txt
# Speichert:
#   - unseen_metrics_cv23_v3.csv
# ------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import joblib


def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom == 0:
        return 0.0
    return (2 * cov) / denom


def main():
    parser = argparse.ArgumentParser(
        description="Evaluiert v3-Tabularmodelle (nur SigMOS+WavLM) auf Unseen-Datensatz."
    )
    parser.add_argument("--unseen_csv", required=True, help="Pfad zur Unseen-CSV (SigMOS+WavLM+WER)")
    parser.add_argument("--models_dir", required=True, help="Ordner mit v3-Modellen und feature_cols_v3.txt")
    parser.add_argument("--out_csv", required=True, help="Pfad zur Ausgabedatei (CSV) mit Unseen-Metriken")
    args = parser.parse_args()

    print(f"Lade Unseen-Daten von: {args.unseen_csv}")
    df = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"Unseen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    targets = ["wer_tiny", "wer_base", "wer_small"]

    # Feature-Liste aus Training laden
    feat_path = os.path.join(args.models_dir, "feature_cols_v3.txt")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"feature_cols_v3.txt nicht gefunden unter: {feat_path}")

    with open(feat_path, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    print(f"Anzahl Feature-Spalten (laut Training): {len(feature_cols)}")

    # Sicherstellen, dass alle Feature-Spalten im Unseen-Set existieren
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Folgende Features fehlen im Unseen-Datensatz: {missing[:20]} ...")

    # Zeilen mit NaN in Targets droppen (falls any)
    before = len(df)
    df = df.dropna(subset=targets)
    after = len(df)
    if after < before:
        print(f"Droppe {before - after} Zeilen mit NaN in Targets. Verbleibend: {after}")

    X_unseen = df[feature_cols].astype(np.float32).values

    results = []

    for target in targets:
        print(f"\n=== Evaluation für Zielvariable: {target} ===")

        y_true = df[target].astype(np.float32).values

        # LightGBM
        lgb_path = os.path.join(args.models_dir, f"LightGBM_v3_{target}.pkl")
        if os.path.exists(lgb_path):
            model_lgb = joblib.load(lgb_path)
            preds_lgb = model_lgb.predict(X_unseen)
            r2 = r2_score(y_true, preds_lgb)
            rmse = mean_squared_error(y_true, preds_lgb, squared=False)
            mae = mean_absolute_error(y_true, preds_lgb)
            ccc = concordance_correlation_coefficient(y_true, preds_lgb)
            results.append({
                "model": "LightGBM_v3",
                "target": target,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "ccc": ccc,
            })
        else:
            print(f"  LightGBM-Modell nicht gefunden: {lgb_path}")

        # CatBoost
        cat_path = os.path.join(args.models_dir, f"CatBoost_v3_{target}.pkl")
        if os.path.exists(cat_path):
            model_cat = joblib.load(cat_path)
            preds_cat = model_cat.predict(X_unseen)
            r2 = r2_score(y_true, preds_cat)
            rmse = mean_squared_error(y_true, preds_cat, squared=False)
            mae = mean_absolute_error(y_true, preds_cat)
            ccc = concordance_correlation_coefficient(y_true, preds_cat)
            results.append({
                "model": "CatBoost_v3",
                "target": target,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "ccc": ccc,
            })
        else:
            print(f"  CatBoost-Modell nicht gefunden: {cat_path}")

    if not results:
        print("Keine Ergebnisse erzeugt – wurden die Modelldateien gefunden?")
        return

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)

    print("\n=== Unseen-Evaluation v3 (nur SigMOS+WavLM) ===")
    print(results_df)
    print(f"\nMetriken gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()