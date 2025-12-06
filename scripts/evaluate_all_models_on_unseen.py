# scripts/evaluate_all_models_on_unseen.py
# -----------------------------------------------------------------------------
# Vergleicht:
#   - Tabular-Modelle (LightGBM + CatBoost) aus CV23-balanced-with-handcrafted
#   - MLP-Multioutput (Baseline)
#   - MLP-Multioutput (getunt, falls ladbar)
#
# auf dem Unseen-Datensatz:
#   - Basis: merged_sigmos_wavlm_unseen.csv
#   - Handcrafted: handcrafted_audio_features_unseen.csv
#
# Metriken: R2, RMSE, MAE, CCC
# -----------------------------------------------------------------------------

import os
import argparse
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------

def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)
    return ccc


def evaluate_regression(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    return r2, rmse, mae, ccc


def get_device(device_arg: str):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


# -----------------------------------------------------------------------------
# Multi-Output-MLP (muss zur Trainings-Architektur passen)
# -----------------------------------------------------------------------------

class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout=0.2, output_dim=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Laden und Mergen der Unseen-Daten
# -----------------------------------------------------------------------------

def load_unseen_with_handcrafted(unseen_csv, handcrafted_csv):
    print(f"Lade Unseen-Basisdaten von: {unseen_csv}")
    df_unseen = pd.read_csv(unseen_csv, low_memory=False)
    print(f"Unseen-Basis: {df_unseen.shape[0]} Zeilen, {df_unseen.shape[1]} Spalten")

    print(f"Lade Handcrafted-Features von: {handcrafted_csv}")
    df_hand = pd.read_csv(handcrafted_csv, low_memory=False)
    print(f"Handcrafted: {df_hand.shape[0]} Zeilen, {df_hand.shape[1]} Spalten")

    # Normierte Filenamen
    df_unseen["filename_norm"] = df_unseen["filename"].astype(str).str.strip().str.lower()
    # Handcrafted hat Spalte "filename"
    df_hand["filename_norm"] = df_hand["filename"].astype(str).str.strip().str.lower()

    df_unseen_merged = pd.merge(
        df_unseen,
        df_hand.drop_duplicates("filename_norm"),
        on="filename_norm",
        how="left",
        suffixes=("", "_hand")
    )

    # Aufräumen
    df_unseen_merged.drop(columns=["filename_norm"], inplace=True)
    # Falls es doppelte "filename" gibt (aus Handcrafted), einen entfernen
    if "filename_hand" in df_unseen_merged.columns:
        df_unseen_merged.drop(columns=["filename_hand"], inplace=True)

    print(f"Nach Merge (Unseen + Handcrafted): {df_unseen_merged.shape[0]} Zeilen, {df_unseen_merged.shape[1]} Spalten")
    return df_unseen_merged


# -----------------------------------------------------------------------------
# Tabular-Modelle evaluieren (LightGBM + CatBoost aus CV23-balanced-with-handcrafted)
# -----------------------------------------------------------------------------

def evaluate_tabular_models(df_unseen, tabular_dir):
    results = []

    # Targets (sind im Unseen-Datensatz vorhanden)
    targets = ["wer_tiny", "wer_base", "wer_small"]

    for target in targets:
        # Dateinamen basierend auf deiner Ordnerstruktur
        lgb_path = os.path.join(tabular_dir, f"LightGBM_with_handcrafted_{target}.pkl")
        cat_path = os.path.join(tabular_dir, f"CatBoost_with_handcrafted_{target}.pkl")

        if not os.path.exists(lgb_path) and not os.path.exists(cat_path):
            print(f"  Keine Tabular-Modelle für {target} gefunden.")
            continue

        # Ground truth
        if target not in df_unseen.columns:
            print(f"  Warnung: Target {target} nicht im Unseen-Datensatz vorhanden. Überspringe.")
            continue
        y_true = df_unseen[target].values

        # Zuerst LightGBM, um Feature-Namen aus dem Modell zu holen
        feature_names = None
        if os.path.exists(lgb_path):
            try:
                lgb_model = joblib.load(lgb_path)
                feature_names = getattr(lgb_model, "feature_name_", None)
            except Exception as e:
                print(f"  Fehler beim Laden von {lgb_path}: {e}")

        # Fallback: wenn keine Feature-Namen im Modell, bestimmen wir sie aus dem Unseen-Datensatz
        if feature_names is None:
            exclude_cols = {
                "filename", "client_id", "age", "gender", "sentence",
                "reference", "hypothesis",
                "wer_tiny", "wer_base", "wer_small"
            }
            feature_names = [
                c for c in df_unseen.columns
                if c not in exclude_cols and np.issubdtype(df_unseen[c].dtype, np.number)
            ]
            print(f"  Konnte keine feature_name_ im Modell lesen, verwende {len(feature_names)} numerische Spalten aus Unseen.")

        # Sicherstellen, dass alle Feature-Spalten im Unseen-Datensatz existieren
        missing = [c for c in feature_names if c not in df_unseen.columns]
        if missing:
            print(f"  Warnung: {len(missing)} Features fehlen im Unseen-Datensatz. Fülle diese mit 0.")
            for c in missing:
                df_unseen[c] = 0.0

        X_unseen = df_unseen[feature_names].astype(np.float32).values

        # LightGBM evaluieren
        if os.path.exists(lgb_path):
            try:
                lgb_model = joblib.load(lgb_path)
                y_pred = lgb_model.predict(X_unseen)
                r2, rmse, mae, ccc = evaluate_regression(y_true, y_pred)
                results.append({
                    "model": "LightGBM_with_handcrafted",
                    "target": target,
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "ccc": ccc
                })
            except Exception as e:
                print(f"  Fehler bei LightGBM-Evaluation für {target}: {e}")

        # CatBoost evaluieren (nutzt dieselben Features)
        if os.path.exists(cat_path):
            try:
                cat_model = joblib.load(cat_path)
                y_pred = cat_model.predict(X_unseen)
                r2, rmse, mae, ccc = evaluate_regression(y_true, y_pred)
                results.append({
                    "model": "CatBoost_with_handcrafted",
                    "target": target,
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "ccc": ccc
                })
            except Exception as e:
                print(f"  Fehler bei CatBoost-Evaluation für {target}: {e}")

    return results


# -----------------------------------------------------------------------------
# MLP-Modelle evaluieren
# -----------------------------------------------------------------------------

def load_mlp_model(checkpoint_path, config_path, input_dim, device):
    """
    Versucht robust, ein MLP-Modell aus checkpoint_path zu laden.
    Unterstützt:
      - torch.save(model, path)
      - torch.save({"model_state_dict": ..., ...}, path)
      - torch.save(model.state_dict(), path)
    Konfiguration (hidden_sizes, dropout) wird aus config_path gelesen.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} existiert nicht.")

    # Konfiguration laden
    hidden_sizes = [512, 256, 128]
    dropout = 0.2
    if config_path is not None and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            if "hidden_sizes" in cfg:
                hidden_sizes = cfg["hidden_sizes"]
            if "dropout" in cfg:
                dropout = cfg["dropout"]
        except Exception as e:
            print(f"  Warnung: Konnte config.json nicht vollständig lesen: {e}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Fall 1: bereits ein nn.Module
    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
        return model

    # Fall 2: dict mit model_state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model = MultiOutputMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout=dropout, output_dim=3).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    # Fall 3: dict, aber ohne expliziten Key -> wir versuchen es als pures state_dict
    if isinstance(ckpt, dict):
        model = MultiOutputMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout=dropout, output_dim=3).to(device)
        model.load_state_dict(ckpt)
        return model

    raise RuntimeError("Unbekanntes Checkpoint-Format für MLP.")


def evaluate_mlp_model(df_unseen, mlp_dir, label, device):
    """
    Evaluierung eines Multi-Output-MLP (wer_tiny, wer_base, wer_small).

    Erwartet in mlp_dir:
      - checkpoint (.pt)
      - scaler.pkl
      - feature_cols.txt
      - optional: config.json
    """
    results = []

    # Targets
    targets = ["wer_tiny", "wer_base", "wer_small"]
    for t in targets:
        if t not in df_unseen.columns:
            print(f"  Warnung: Target {t} nicht im Unseen-Datensatz, MLP-Eval für {label} wird übersprungen.")
            return results

    # Feature-Liste laden
    feat_path = os.path.join(mlp_dir, "feature_cols.txt")
    if not os.path.exists(feat_path):
        print(f"  Feature-Liste {feat_path} nicht gefunden, MLP-Evaluation ({label}) übersprungen.")
        return results

    with open(feat_path, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # Sicherstellen, dass alle Feature-Spalten existieren
    missing = [c for c in feature_cols if c not in df_unseen.columns]
    if missing:
        print(f"  Warnung: {len(missing)} Feature-Spalten fehlen im Unseen-Datensatz für {label}. Fülle mit 0.")
        for c in missing:
            df_unseen[c] = 0.0

    X_unseen = df_unseen[feature_cols].astype(np.float32).values

    # Targets
    y_true = df_unseen[targets].astype(np.float32).values

    # Scaler laden
    scaler_path = os.path.join(mlp_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print(f"  Scaler {scaler_path} nicht gefunden, MLP-Evaluation ({label}) übersprungen.")
        return results

    scaler = joblib.load(scaler_path)
    X_unseen_scaled = scaler.transform(X_unseen)

    # Modell laden
    config_path = os.path.join(mlp_dir, "config.json")
    input_dim = X_unseen.shape[1]

    # Checkpoint-Datei bestimmen
    # Für Baseline erwarten wir "best_model.pt"
    # Für getuntes Modell meist "mlp_multioutput_best.pt"
    ckpt_candidates = [
        os.path.join(mlp_dir, "best_model.pt"),
        os.path.join(mlp_dir, "mlp_multioutput_best.pt")
    ]
    ckpt_path = None
    for cand in ckpt_candidates:
        if os.path.exists(cand):
            ckpt_path = cand
            break

    if ckpt_path is None:
        print(f"  Kein MLP-Checkpoint in {mlp_dir} gefunden, MLP-Evaluation ({label}) übersprungen.")
        return results

    try:
        model = load_mlp_model(ckpt_path, config_path, input_dim=input_dim, device=device)
    except Exception as e:
        print(f"  Fehler beim Laden des MLP-Checkpoints ({label}): {e}")
        return results

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_unseen_scaled, dtype=torch.float32, device=device)
        y_pred = model(X_tensor).cpu().numpy()

    # Pro Target Metriken
    for i, target in enumerate(targets):
        r2, rmse, mae, ccc = evaluate_regression(y_true[:, i], y_pred[:, i])
        results.append({
            "model": label,
            "target": target,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "ccc": ccc
        })

    return results


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluierung aller Modelle auf Unseen-Daten")
    parser.add_argument("--unseen_csv", required=True, help="Pfad zur Unseen-Basis-CSV (merged_sigmos_wavlm_unseen.csv)")
    parser.add_argument("--handcrafted_unseen_csv", required=True, help="Pfad zu handcrafted_features_unseen.csv")
    parser.add_argument("--tabular_dir", required=True, help="Verzeichnis mit Tabular-Modellen (CV23 balanced + handcrafted)")
    parser.add_argument("--mlp_dir", required=True, help="Verzeichnis mit MLP-Baseline (multioutput)")
    parser.add_argument("--mlp_tuned_dir", required=True, help="Verzeichnis mit MLP getunt (fullgrid)")
    parser.add_argument("--out_csv", required=True, help="Pfad zur Ergebnis-CSV")
    parser.add_argument("--device", default="auto", help="Gerät: 'auto', 'cpu', 'cuda' oder 'mps'")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Verwende Gerät: {device}")

    # Daten laden
    df_unseen = load_unseen_with_handcrafted(args.unseen_csv, args.handcrafted_unseen_csv)

    all_results = []

    # 1) Tabular-Modelle
    print("\n=== Evaluation Tabular (LightGBM + CatBoost, mit Handcrafted) ===")
    tab_results = evaluate_tabular_models(df_unseen.copy(), args.tabular_dir)
    all_results.extend(tab_results)

    # 2) MLP-Baseline
    print("\n=== Evaluation MLP (Baseline) ===")
    mlp_base_results = evaluate_mlp_model(df_unseen.copy(), args.mlp_dir, label="MLP_multioutput_baseline", device=device)
    all_results.extend(mlp_base_results)

    # 3) MLP getunt
    print("\n=== Evaluation MLP (getunt, full grid) ===")
    mlp_tuned_results = evaluate_mlp_model(df_unseen.copy(), args.mlp_tuned_dir, label="MLP_multioutput_tuned", device=device)
    all_results.extend(mlp_tuned_results)

    if not all_results:
        print("Keine Ergebnisse erzeugt – vermutlich konnten keine Modelle geladen werden.")
        return

    results_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)

    print("\n=== Gesamtergebnisse (Unseen) ===")
    print(results_df)

    print(f"\nErgebnisse gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()