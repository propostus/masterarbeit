# scripts/evaluate_cv23_handcrafted_on_unseen.py
# ------------------------------------------------------
# Evaluierung der CV23-Tabularmodelle (mit Handcrafted-Features)
# auf dem Unseen-Datensatz (SigMOS+WavLM+Handcrafted).
#
# Erwartet:
#   - unseen_base_csv:  merged_sigmos_wavlm_unseen.csv
#   - unseen_handcrafted_csv: handcrafted_audio_features_unseen.csv
#   - models_dir:  results/model_exports_cv23_balanced_with_handcrafted
#
# Lädt:
#   LightGBM_with_handcrafted_<target>.pkl
#   CatBoost_with_handcrafted_<target>.pkl
#
# Targets:
#   wer_tiny, wer_base, wer_small
# ------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import joblib


def concordance_correlation_coefficient(y_true, y_pred):
    """CCC nach Lin (1989)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)
    return ccc


def build_feature_matrix(df: pd.DataFrame) -> (np.ndarray, list):
    """
    Erzeugt Feature-Matrix X aus df, analog zur Trainingslogik:
    - alle numerischen Spalten
    - bestimmte Spalten explizit ausgeschlossen
    """
    exclude_cols = [
        "filename",
        "client_id",
        "age",
        "gender",
        "sentence",
        "reference",
        "hypothesis",
        "wer_tiny",
        "wer_base",
        "wer_small",
    ]

    feature_cols = [
        c
        for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    X = df[feature_cols].astype(np.float32).values
    return X, feature_cols


def evaluate_model(model, X, y_true):
    """Berechnet R², RMSE, MAE, CCC für ein gegebenes Modell."""
    y_pred = model.predict(X)
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    return r2, rmse, mae, ccc


def main():
    parser = argparse.ArgumentParser(
        description="Evaluierung der CV23-Tabularmodelle (mit Handcrafted-Features) auf Unseen-Daten"
    )
    parser.add_argument(
        "--unseen_base_csv",
        required=True,
        help="Pfad zur Unseen-Basis-CSV (merged_sigmos_wavlm_unseen.csv)",
    )
    parser.add_argument(
        "--unseen_handcrafted_csv",
        required=True,
        help="Pfad zur Unseen-Handcrafted-CSV (handcrafted_audio_features_unseen.csv)",
    )
    parser.add_argument(
        "--models_dir",
        required=True,
        help="Verzeichnis mit den trainierten Modellen (cv23_balanced_with_handcrafted)",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Pfad zur Ergebnis-CSV für die Unseen-Evaluation",
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # 1. Unseen-Basis laden
    # --------------------------------------------------
    print(f"Lade Unseen-Basisdaten von: {args.unseen_base_csv}")
    df_base = pd.read_csv(args.unseen_base_csv, low_memory=False)
    print(f"Unseen-Basis: {df_base.shape[0]} Zeilen, {df_base.shape[1]} Spalten")

    # Sicherstellen, dass 'filename' existiert
    if "filename" not in df_base.columns:
        raise RuntimeError("In der Unseen-Basis fehlt die Spalte 'filename'.")

    # --------------------------------------------------
    # 2. Handcrafted-Features laden und mergen
    # --------------------------------------------------
    print(f"Lade Unseen-Handcrafted von: {args.unseen_handcrafted_csv}")
    df_hand = pd.read_csv(args.unseen_handcrafted_csv, low_memory=False)

    # Erwartete Spalte: filename
    if "filename" not in df_hand.columns:
        raise RuntimeError("In der Handcrafted-CSV fehlt die Spalte 'filename'.")

    print(f"Handcrafted: {df_hand.shape[0]} Zeilen, {df_hand.shape[1]} Spalten")

    # Merge (inner join auf filename)
    df_merged = pd.merge(df_base, df_hand, on="filename", how="inner")
    print(
        f"Nach Merge (Unseen+Handcrafted): {df_merged.shape[0]} Zeilen, "
        f"{df_merged.shape[1]} Spalten"
    )

    # Optional: Zeilen ohne Targets entfernen
    before = df_merged.shape[0]
    df_merged = df_merged.dropna(subset=["wer_tiny", "wer_base", "wer_small"])
    after = df_merged.shape[0]
    if after < before:
        print(f"{before - after} Zeilen ohne gültige WER-Targets entfernt.")
    print(f"Finaler Unseen-Merge: {df_merged.shape[0]} Zeilen")

    # --------------------------------------------------
    # 3. Feature-Matrix X, Targets y
    # --------------------------------------------------
    X_unseen, feature_cols = build_feature_matrix(df_merged)
    print(f"Feature-Dimension (Unseen+Handcrafted): {X_unseen.shape[1]}")

    targets = ["wer_tiny", "wer_base", "wer_small"]

    # --------------------------------------------------
    # 4. Modelle laden und evaluieren
    # --------------------------------------------------
    results = []

    for tgt in targets:
        print(f"\n=== Evaluierung für Zielvariable: {tgt} ===")
        y_unseen = df_merged[tgt].astype(np.float32).values

        # LightGBM
        lgb_path = os.path.join(
            args.models_dir, f"LightGBM_with_handcrafted_{tgt}.pkl"
        )
        if os.path.exists(lgb_path):
            print(f"→ Lade LightGBM: {lgb_path}")
            lgb_model = joblib.load(lgb_path)
            try:
                r2, rmse, mae, ccc = evaluate_model(lgb_model, X_unseen, y_unseen)
                results.append(
                    {
                        "model": "LightGBM_with_handcrafted",
                        "target": tgt,
                        "r2": r2,
                        "rmse": rmse,
                        "mae": mae,
                        "ccc": ccc,
                    }
                )
            except Exception as e:
                print(f"  Fehler bei LightGBM ({tgt}): {e}")
        else:
            print(f"  Kein LightGBM-Modell gefunden unter {lgb_path}")

        # CatBoost
        cat_path = os.path.join(
            args.models_dir, f"CatBoost_with_handcrafted_{tgt}.pkl"
        )
        if os.path.exists(cat_path):
            print(f"→ Lade CatBoost: {cat_path}")
            cat_model = joblib.load(cat_path)
            try:
                r2, rmse, mae, ccc = evaluate_model(cat_model, X_unseen, y_unseen)
                results.append(
                    {
                        "model": "CatBoost_with_handcrafted",
                        "target": tgt,
                        "r2": r2,
                        "rmse": rmse,
                        "mae": mae,
                        "ccc": ccc,
                    }
                )
            except Exception as e:
                print(f"  Fehler bei CatBoost ({tgt}): {e}")
        else:
            print(f"  Kein CatBoost-Modell gefunden unter {cat_path}")

    if not results:
        print("Keine Ergebnisse erzeugt. Prüfe Pfade und Modellspeicher.")
        return

    res_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    res_df.to_csv(args.out_csv, index=False)

    print("\n=== Unseen-Evaluation (CV23 + Handcrafted) ===")
    print(res_df)
    print(f"\nMetriken gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()