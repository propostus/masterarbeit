# scripts/evaluate_all_tabular_on_unseen.py
import os
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

import joblib
from catboost import CatBoostRegressor


TARGETS = ["wer_tiny", "wer_base", "wer_small"]


def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = [
        "filename",
        "file",
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
    exclude = [c for c in exclude if c in df.columns]
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and np.issubdtype(df[c].dtype, np.number)
    ]
    return feature_cols


def eval_family(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: str,
    family_name: str,
    lgb_pattern: str,
    cat_pattern: str,
) -> list[dict]:
    if models_dir is None or not os.path.isdir(models_dir):
        print(f"→ Ordner für {family_name} nicht gefunden, überspringe.")
        return []

    X = df[feature_cols].astype(np.float32).values
    results = []

    for target in TARGETS:
        if target not in df.columns:
            print(f"  {family_name}: Spalte {target} nicht im Datensatz, überspringe.")
            continue
        y = df[target].values

        # LightGBM
        lgb_path = os.path.join(models_dir, lgb_pattern.format(target=target))
        if os.path.exists(lgb_path):
            try:
                model = joblib.load(lgb_path)
                preds = model.predict(X)
                r2 = r2_score(y, preds)
                rmse = mean_squared_error(y, preds, squared=False)
                mae = mean_absolute_error(y, preds)
                ccc = concordance_correlation_coefficient(y, preds)
                results.append(
                    {
                        "family": family_name,
                        "model": "LightGBM",
                        "target": target,
                        "r2": r2,
                        "rmse": rmse,
                        "mae": mae,
                        "ccc": ccc,
                    }
                )
            except Exception as e:
                print(f"  Fehler LightGBM ({family_name}, {target}): {e}")

        # CatBoost
        cat_path = os.path.join(models_dir, cat_pattern.format(target=target))
        if os.path.exists(cat_path):
            try:
                model = CatBoostRegressor()
                model.load_model(cat_path)
                preds = model.predict(X)
                r2 = r2_score(y, preds)
                rmse = mean_squared_error(y, preds, squared=False)
                mae = mean_absolute_error(y, preds)
                ccc = concordance_correlation_coefficient(y, preds)
                results.append(
                    {
                        "family": family_name,
                        "model": "CatBoost",
                        "target": target,
                        "r2": r2,
                        "rmse": rmse,
                        "mae": mae,
                        "ccc": ccc,
                    }
                )
            except Exception as e:
                print(f"  Fehler CatBoost ({family_name}, {target}): {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluiert CV23-Tabular-Modelle (LightGBM / CatBoost) auf Unseen-Daten."
    )
    parser.add_argument(
        "--unseen_csv",
        required=True,
        help="Unseen-CSV mit SigMOS+WavLM+WER (z.B. merged_sigmos_wavlm_unseen.csv)",
    )
    parser.add_argument(
        "--unseen_extra_features_csv",
        type=str,
        default=None,
        help="Optional: zusätzliche Audiofeatures für Unseen (für Handcrafted-Modelle)",
    )
    parser.add_argument(
        "--speaker_models_dir",
        required=True,
        help="Ordner mit Modellen aus speakerlevel_full (lightgbm_wer_tiny.pkl, catboost_wer_tiny.cbm, ...)",
    )
    parser.add_argument(
        "--handcrafted_models_dir",
        type=str,
        default=None,
        help="Ordner mit Modellen aus balanced_with_handcrafted (LightGBM_with_handcrafted_wer_tiny.pkl, ...)",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Pfad zur Ergebnis-CSV",
    )
    args = parser.parse_args()

    # 1) Unseen-Basisdaten laden
    print(f"Lade Unseen-Basisdaten von: {args.unseen_csv}")
    df_unseen_base = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"Unseen-Basis: {df_unseen_base.shape[0]} Zeilen, {df_unseen_base.shape[1]} Spalten")

    results = []

    # 2) Speakerlevel_full-Modelle (nur SigMOS+WavLM)
    print("\n=== Evaluation: speakerlevel_full-Modelle (nur SigMOS+WavLM) ===")
    feat_base = get_feature_cols(df_unseen_base)
    print(f"Feature-Dimension (Basis): {len(feat_base)}")
    results.extend(
        eval_family(
            df_unseen_base,
            feat_base,
            models_dir=args.speaker_models_dir,
            family_name="speakerlevel_full",
            lgb_pattern="lightgbm_{target}.pkl",
            cat_pattern="catboost_{target}.cbm",
        )
    )

    # 3) Handcrafted-Modelle (nur wenn Extra-Features vorhanden)
    if args.handcrafted_models_dir is not None and os.path.isdir(args.handcrafted_models_dir):
        if args.unseen_extra_features_csv is None:
            print(
                "\n=== Handcrafted-Modelle: keine unseen_extra_features_csv angegeben, "
                "überspringe diese Familie. ==="
            )
        elif not os.path.exists(args.unseen_extra_features_csv):
            print(
                f"\n=== Handcrafted-Modelle: Extra-Feature-Datei {args.unseen_extra_features_csv} "
                "nicht gefunden, überspringe. ==="
            )
        else:
            print("\n=== Evaluation: balanced_with_handcrafted-Modelle (mit Extra-Features) ===")
            df_extra = pd.read_csv(args.unseen_extra_features_csv, low_memory=False)

            # robustes Join auf filename/file
            base_key = "filename" if "filename" in df_unseen_base.columns else "file"
            extra_key = "filename" if "filename" in df_extra.columns else "file"

            df_merged = df_unseen_base.merge(
                df_extra, left_on=base_key, right_on=extra_key, how="inner", suffixes=("", "_extra")
            )
            print(f"Unseen + Handcrafted: {df_merged.shape[0]} Zeilen, {df_merged.shape[1]} Spalten")

            feat_extra = get_feature_cols(df_merged)
            print(f"Feature-Dimension (mit Handcrafted): {len(feat_extra)}")

            results.extend(
                eval_family(
                    df_merged,
                    feat_extra,
                    models_dir=args.handcrafted_models_dir,
                    family_name="balanced_with_handcrafted",
                    lgb_pattern="LightGBM_with_handcrafted_{target}.pkl",
                    cat_pattern="CatBoost_with_handcrafted_{target}.cbm",
                )
            )

    # 4) Ergebnisse speichern
    if results:
        df_res = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df_res.to_csv(args.out_csv, index=False)
        print("\n=== Unseen-Evaluation abgeschlossen ===")
        print(df_res)
        print(f"\nErgebnisse gespeichert unter: {args.out_csv}")
    else:
        print("Keine Ergebnisse erzeugt (evtl. keine Modelle gefunden?).")


if __name__ == "__main__":
    main()