# scripts/eval_cv23_mlp_multioutput_v3_on_unseen.py
# ------------------------------------------------------
# Lädt das MLP v3 (nur SigMOS+WavLM) und evaluiert auf dem
# Unseen-Datensatz (merged_sigmos_wavlm_unseen.csv).
# Erwartet im model_dir:
#   - mlp_multioutput_v3_best.pt
#   - scaler_mlp_v3.pkl
#   - feature_cols_mlp_v3.txt
#   - config_mlp_v3.json
# Berechnet R², RMSE, MAE, CCC je Zielvariable.
# ------------------------------------------------------

import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn


def select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA nicht verfügbar, verwende CPU.")
        return torch.device("cpu")
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS nicht verfügbar, verwende CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


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


class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluiert MLP v3 (nur SigMOS+WavLM) auf Unseen-Datensatz."
    )
    parser.add_argument("--unseen_csv", required=True, help="Pfad zur Unseen-CSV (SigMOS+WavLM+WER)")
    parser.add_argument("--model_dir", required=True, help="Ordner mit MLP v3 Checkpoint und Scaler")
    parser.add_argument("--out_csv", required=True, help="Pfad zur Ausgabedatei (CSV) mit Unseen-Metriken")
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Verwende Gerät: {device}")

    print(f"Lade Unseen-Daten von: {args.unseen_csv}")
    df = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"Unseen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    targets = ["wer_tiny", "wer_base", "wer_small"]

    feature_path = os.path.join(args.model_dir, "feature_cols_mlp_v3.txt")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature-Datei nicht gefunden: {feature_path}")
    with open(feature_path, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    print(f"Anzahl Feature-Spalten (MLP v3): {len(feature_cols)}")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Folgende Features fehlen im Unseen-Datensatz: {missing[:20]} ...")

    before = len(df)
    df = df.dropna(subset=targets)
    after = len(df)
    if after < before:
        print(f"Droppe {before - after} Zeilen mit NaN in Targets. Verbleibend: {after}")

    X_unseen = df[feature_cols].astype(np.float32).values
    y_unseen = df[targets].astype(np.float32).values

    import joblib
    scaler_path = os.path.join(args.model_dir, "scaler_mlp_v3.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler-Datei nicht gefunden: {scaler_path}")
    scaler: StandardScaler = joblib.load(scaler_path)
    X_unseen_scaled = scaler.transform(X_unseen)

    config_path = os.path.join(args.model_dir, "config_mlp_v3.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    input_dim = config["input_dim"]
    hidden_sizes = config["hidden_sizes"]
    dropout = config["dropout"]

    model = MultiOutputMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout=dropout).to(device)

    ckpt_path = os.path.join(args.model_dir, "mlp_multioutput_v3_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MLP-Checkpoint nicht gefunden: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    X_tensor = torch.tensor(X_unseen_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    results = []
    for i, target in enumerate(targets):
        y_true = y_unseen[:, i]
        y_pred = preds[:, i]
        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        ccc = concordance_correlation_coefficient(y_true, y_pred)
        results.append(
            {
                "model": "MLP_multioutput_v3",
                "target": target,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "ccc": ccc,
            }
        )

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)

    print("\n=== Unseen-Evaluation (MLP v3, nur SigMOS+WavLM) ===")
    print(results_df)
    print(f"\nMetriken gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()