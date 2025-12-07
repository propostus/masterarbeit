# src/regression.py
import os
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")  # optional

class MultiOutputMLP(nn.Module):
    """Muss zur Trainings-Architektur passen."""
    def __init__(self, input_dim, hidden_sizes, output_dim=3, dropout=0.2):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def load_regression_model(
    models_dir: str = "models/regression",
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Lädt:
      - feature_cols.txt
      - scaler.pkl
      - mlp_regression_multiwer.pt
      - mlp_regression_config_and_metrics.json

    Rückgabe:
      {
        "device": torch.device,
        "feature_cols": List[str],
        "scaler": StandardScaler,
        "model": MultiOutputMLP,
        "config": dict
      }
    """
    dev = _resolve_device(device)

    # Feature-Liste
    feature_cols_path = os.path.join(models_dir, "feature_cols.txt")
    with open(feature_cols_path, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # Scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Config (für hidden_sizes + dropout)
    config_path = os.path.join(models_dir, "mlp_regression_config_and_metrics.json")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    best_cfg = config_data["best_config"]
    hidden_sizes = best_cfg["hidden_sizes"]
    dropout = best_cfg.get("dropout", 0.2)

    # Modell
    input_dim = len(feature_cols)
    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        output_dim=3,
        dropout=dropout,
    ).to(dev)

    weights_path = os.path.join(models_dir, "mlp_regression_multiwer.pt")
    state = torch.load(weights_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()

    return {
        "device": dev,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "model": model,
        "config": config_data,
    }


def predict_wer_from_features(
    df_features: pd.DataFrame,
    bundle: Dict[str, Any],
) -> pd.DataFrame:
    """
    df_features: DataFrame mit mindestens bundle['feature_cols'].
    bundle: Rückgabe von load_regression_model.

    Rückgabe: DataFrame mit filename + wer_tiny_pred/base_pred/small_pred.
    """
    feature_cols = bundle["feature_cols"]
    scaler = bundle["scaler"]
    model: MultiOutputMLP = bundle["model"]
    dev: torch.device = bundle["device"]

    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise ValueError(f"Fehlende Feature-Spalten in df_features: {missing}")

    X = df_features[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)

    x_tensor = torch.from_numpy(X_scaled).to(dev)
    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy()  # (N,3)

    out = pd.DataFrame()
    if "filename" in df_features.columns:
        out["filename"] = df_features["filename"]
    elif "file" in df_features.columns:
        out["filename"] = df_features["file"]

    out["wer_tiny_pred"] = preds[:, 0]
    out["wer_base_pred"] = preds[:, 1]
    out["wer_small_pred"] = preds[:, 2]
    return out