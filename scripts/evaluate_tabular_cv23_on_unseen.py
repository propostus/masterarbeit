# scripts/evaluate_tabular_cv23_on_unseen.py
# ------------------------------------------------------
# Evaluierung von LightGBM- und CatBoost-Modellen
# (trainiert auf CV23-balanced + Handcrafted-Features)
# auf dem Unseen-Datensatz.
#
# WICHTIG:
#   - Die Feature-Reihenfolge wird aus dem TRAINING
#     rekonstruiert (CV23-balanced + Handcrafted),
#     damit sie exakt mit den Modellen übereinstimmt.
#   - Modelle werden flexibel gefunden:
#       * Dateiname enthält Zielnamen (wer_tiny/base/small)
#       * LightGBM: .pkl/.joblib und "light"/"lgb" im Namen
#       * CatBoost: .cbm oder "catboost" im Namen
# ------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor
import joblib


def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)
    return float(ccc)


# ------------------------------------------------------
# Feature-Liste aus dem TRAINING rekonstruieren
# ------------------------------------------------------
def get_training_feature_columns(train_dataset_csv, train_handcrafted_csv):
    """
    Rekonstruiert die Spaltenreihenfolge, wie sie beim Training verwendet wurde:
    - Basis-CSV (CV23-balanced Multi-WER)
    - + Handcrafted-CSV (CV23-balanced)
    - Dann: alle Spalten außer den bekannten Meta-/Target-Spalten.
    """

    # Nur Kopf einlesen, um Spalten zu bekommen (kein voller Speicherbedarf)
    base_header = pd.read_csv(train_dataset_csv, nrows=0)
    hc_header = pd.read_csv(train_handcrafted_csv, nrows=0)

    base_cols = list(base_header.columns)
    hc_cols = list(hc_header.columns)

    base_fname_col = "filename" if "filename" in base_cols else "file"
    hc_fname_col = "filename" if "filename" in hc_cols else "file"

    # So merged pandas beim Training: key-Spalte + alle restlichen Base-Spalten + alle HC-Spalten
    merged_cols = (
        ["filename"]
        + [c for c in base_cols if c != base_fname_col]
        + [c for c in hc_cols if c != hc_fname_col]
    )

    # welche Spalten sollen NICHT als Features verwendet werden?
    exclude_cols = {
        "filename",
        "client_id",
        "age",
        "gender",
        "sentence",
        "segment",
        "sentence_id",
        "sentence_domain",
        "locale",
        "variant",
        "accents",
        "up_votes",
        "down_votes",
        "reference",
        "hypothesis",
        "wer_tiny",
        "wer_base",
        "wer_small",
    }

    feature_cols = [c for c in merged_cols if c not in exclude_cols]

    print(f"Anzahl Feature-Spalten im TRAINING: {len(feature_cols)}")
    return feature_cols


# ------------------------------------------------------
# Unseen + Handcrafted laden und mergen
# ------------------------------------------------------
def load_unseen_with_handcrafted(unseen_csv, handcrafted_csv):
    print(f"Lade Unseen-Basisdaten von: {unseen_csv}")
    df_base = pd.read_csv(unseen_csv, low_memory=False)

    base_fname_col = "filename" if "filename" in df_base.columns else "file"
    df_base.rename(columns={base_fname_col: "filename"}, inplace=True)
    print(f"Unseen-Basis: {df_base.shape[0]} Zeilen, {df_base.shape[1]} Spalten")

    print(f"Lade Handcrafted-Features von: {handcrafted_csv}")
    df_hc = pd.read_csv(handcrafted_csv, low_memory=False)
    hc_fname_col = "filename" if "filename" in df_hc.columns else "file"
    df_hc.rename(columns={hc_fname_col: "filename"}, inplace=True)
    print(f"Handcrafted: {df_hc.shape[0]} Zeilen, {df_hc.shape[1]} Spalten")

    df = df_base.merge(df_hc, on="filename", how="inner")
    print(f"Nach Merge (Unseen + Handcrafted): {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    for col in ["wer_tiny", "wer_base", "wer_small"]:
        if col not in df.columns:
            raise RuntimeError(f"Spalte '{col}' fehlt im gemergten Unseen-DataFrame.")

    before = df.shape[0]
    df = df.dropna(subset=["wer_tiny", "wer_base", "wer_small"])
    after = df.shape[0]
    if after < before:
        print(f"Warnung: {before - after} Zeilen wegen fehlender Targets entfernt.")

    return df


def build_feature_matrix_from_training_order(df_unseen, feature_cols_train):
    missing = [c for c in feature_cols_train if c not in df_unseen.columns]
    if missing:
        raise RuntimeError(
            f"Folgende Trainings-Features fehlen im Unseen-DataFrame: {missing[:10]} "
            f"(insgesamt {len(missing)})"
        )

    X = df_unseen[feature_cols_train].astype(np.float32).values
    print(f"Anzahl Feature-Spalten (Unseen, nach Trainings-Order): {X.shape[1]}")
    return X


# ------------------------------------------------------
# Modelldatei finden und Evaluation
# ------------------------------------------------------
def evaluate_model(model, X, y, model_label, target_name):
    preds = model.predict(X)
    preds = np.asarray(preds).reshape(-1)

    r2 = r2_score(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)
    ccc = concordance_correlation_coefficient(y, preds)

    return {
        "model": model_label,
        "target": target_name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "ccc": ccc,
    }


def find_model_file(files, target, kind):
    target_low = target.lower()
    candidates = []

    for f in files:
        name = f.lower()
        if target_low not in name:
            continue

        if kind == "lightgbm":
            if (name.endswith(".pkl") or name.endswith(".joblib")) and (
                "light" in name or "lgb" in name
            ):
                candidates.append(f)
        elif kind == "catboost":
            if name.endswith(".cbm") or "catboost" in name:
                candidates.append(f)

    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        print(
            f"Warnung: mehrere {kind}-Kandidaten für {target}: {candidates}. "
            f"Verwende {candidates[0]}"
        )
    return candidates[0]


# ------------------------------------------------------
# main
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluiert CV23-Tabularmodelle (LightGBM/CatBoost) auf dem Unseen-Datensatz."
    )
    parser.add_argument(
        "--unseen_csv",
        type=str,
        required=True,
        help="Pfad zur Unseen-Basis-CSV (merged_sigmos_wavlm_unseen.csv)",
    )
    parser.add_argument(
        "--handcrafted_unseen_csv",
        type=str,
        required=True,
        help="Pfad zur CSV mit Handcrafted-Features für Unseen",
    )
    parser.add_argument(
        "--train_dataset_csv",
        type=str,
        required=True,
        help="Pfad zur TRAININGS-Basis-CSV (merged_sigmos_wavlm_cv23_balanced_multiwer.csv)",
    )
    parser.add_argument(
        "--train_handcrafted_csv",
        type=str,
        required=True,
        help="Pfad zur TRAININGS-Handcrafted-CSV (handcrafted_audio_features_cv23_balanced.csv)",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Verzeichnis mit den trainierten Tabular-Modellen",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Pfad zur Ausgabedatei mit den Metriken",
    )

    args = parser.parse_args()

    # 1) Trainings-Featureliste rekonstruieren
    feature_cols_train = get_training_feature_columns(
        args.train_dataset_csv, args.train_handcrafted_csv
    )

    # 2) Unseen + Handcrafted laden und nach Trainings-Order aufbauen
    df_unseen = load_unseen_with_handcrafted(
        args.unseen_csv, args.handcrafted_unseen_csv
    )
    X_unseen = build_feature_matrix_from_training_order(
        df_unseen, feature_cols_train
    )

    files_in_models_dir = sorted(os.listdir(args.models_dir))
    print(f"\nModelldateien in {args.models_dir}:")
    for f in files_in_models_dir:
        print("  ", f)

    results = []
    targets = ["wer_tiny", "wer_base", "wer_small"]

    for tgt in targets:
        print(f"\n=== Evaluation für Zielvariable: {tgt} ===")
        y_unseen = df_unseen[tgt].values.astype(np.float32)

        # LightGBM
        lgb_file = find_model_file(files_in_models_dir, tgt, kind="lightgbm")
        if lgb_file is not None:
            lgb_path = os.path.join(args.models_dir, lgb_file)
            try:
                lgb_model = joblib.load(lgb_path)
                res_lgb = evaluate_model(
                    lgb_model,
                    X_unseen,
                    y_unseen,
                    model_label="LightGBM_with_handcrafted",
                    target_name=tgt,
                )
                results.append(res_lgb)
                print(
                    f"LightGBM {tgt}: R2={res_lgb['r2']:.4f}, "
                    f"RMSE={res_lgb['rmse']:.4f}, MAE={res_lgb['mae']:.4f}, "
                    f"CCC={res_lgb['ccc']:.4f}"
                )
            except Exception as e:
                print(f"Fehler bei LightGBM ({tgt}): {e}")
        else:
            print(f"Kein LightGBM-Modell für {tgt} gefunden.")

        # CatBoost
        cb_file = find_model_file(files_in_models_dir, tgt, kind="catboost")
        if cb_file is not None:
            cb_path = os.path.join(args.models_dir, cb_file)
            try:
                cb_model = CatBoostRegressor()
                cb_model.load_model(cb_path)
                res_cb = evaluate_model(
                    cb_model,
                    X_unseen,
                    y_unseen,
                    model_label="CatBoost_with_handcrafted",
                    target_name=tgt,
                )
                results.append(res_cb)
                print(
                    f"CatBoost {tgt}: R2={res_cb['r2']:.4f}, "
                    f"RMSE={res_cb['rmse']:.4f}, MAE={res_cb['mae']:.4f}, "
                    f"CCC={res_cb['ccc']:.4f}"
                )
            except Exception as e:
                print(f"Fehler bei CatBoost ({tgt}): {e}")
        else:
            print(f"Kein CatBoost-Modell für {tgt} gefunden.")

    if not results:
        print("Keine Ergebnisse erzeugt – wurden die Modelldateien gefunden?")
        return

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)

    print("\n=== Fertig. Unseen-Metriken ===")
    print(results_df)
    print(f"\nGespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()