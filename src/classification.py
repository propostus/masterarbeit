# src/classification.py
import os
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

# beste Hyperparameter pro (target, threshold) – aus deiner Tabelle
BEST_CONFIGS = {
    ("wer_tiny", 0.05): {"hidden_sizes": [512, 256], "dropout": 0.3},
    ("wer_tiny", 0.10): {"hidden_sizes": [512, 256], "dropout": 0.3},
    ("wer_tiny", 0.20): {"hidden_sizes": [512, 256], "dropout": 0.3},
    ("wer_base", 0.05): {"hidden_sizes": [512, 256], "dropout": 0.3},
    ("wer_base", 0.10): {"hidden_sizes": [512, 256], "dropout": 0.3},
    ("wer_base", 0.20): {"hidden_sizes": [512, 256, 128], "dropout": 0.3},
    ("wer_small", 0.05): {"hidden_sizes": [512, 256], "dropout": 0.3},
    ("wer_small", 0.10): {"hidden_sizes": [512, 256], "dropout": 0.2},
    ("wer_small", 0.20): {"hidden_sizes": [512, 256], "dropout": 0.3},
}

# optimierte Probability-Thresholds – aus deiner Tabelle
PROB_THRESHOLDS = {
    ("wer_tiny", 0.05): 0.85,
    ("wer_tiny", 0.10): 0.75,
    ("wer_tiny", 0.20): 0.55,
    ("wer_base", 0.05): 0.65,
    ("wer_base", 0.10): 0.55,
    ("wer_base", 0.20): 0.30,
    ("wer_small", 0.05): 0.35,
    ("wer_small", 0.10): 0.25,
    ("wer_small", 0.20): 0.10,
}


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # Binary-Logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,)


def _resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _format_thr_tag(thr: float) -> str:
    """0.05 -> '005', 0.10 -> '010', 0.2 -> '020'."""
    return f"{int(round(thr * 100)):03d}"


def load_feature_cols(feature_cols_path: str = "models/regression/feature_cols.txt"):
    with open(feature_cols_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_binary_classifier(
    target: str,
    threshold: float,
    models_dir: str = "models/classification",
    feature_cols_path: str = "models/regression/feature_cols.txt",
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Lädt ein Binary-MLP-Modell + Scaler + Feature-Liste für eine gegebene WER-Zielvariable
    und eine WER-Schwelle.

    target: 'wer_tiny', 'wer_base' oder 'wer_small'
    threshold: 0.05, 0.10 oder 0.20
    """
    dev = _resolve_device(device)
    thr = float(threshold)
    key = (target, thr)

    if key not in BEST_CONFIGS:
        raise ValueError(f"Keine Konfiguration für {key} in BEST_CONFIGS.")

    cfg = BEST_CONFIGS[key]
    hidden_sizes = cfg["hidden_sizes"]
    dropout = cfg["dropout"]

    feature_cols = load_feature_cols(feature_cols_path)
    input_dim = len(feature_cols)

    thr_tag = _format_thr_tag(thr)
    base_name = f"mlp_binary_{target}_thr{thr_tag}"

    # Modell laden
    model = SmallMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout=dropout).to(dev)
    weights_path = os.path.join(models_dir, base_name + ".pt")
    state = torch.load(weights_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()

    # Scaler laden (modell-spezifisch)
    scaler_path = os.path.join(models_dir, base_name + "_scaler.pkl")
    scaler = joblib.load(scaler_path)

    prob_thr = PROB_THRESHOLDS.get(key, 0.5)

    return {
        "device": dev,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "model": model,
        "target": target,
        "wer_threshold": thr,
        "prob_threshold": prob_thr,
    }


def predict_quality_from_features(
    df_features: pd.DataFrame,
    bundle: Dict[str, Any],
) -> pd.DataFrame:
    """
    df_features: DataFrame mit mindestens bundle['feature_cols'].
    bundle: Rückgabe von load_binary_classifier.

    Rückgabe: DataFrame mit:
      - filename
      - prob_good (p = P(WER <= threshold))
      - label_good (0/1)
    """
    feature_cols = bundle["feature_cols"]
    scaler = bundle["scaler"]
    model: SmallMLP = bundle["model"]
    dev: torch.device = bundle["device"]
    prob_thr: float = bundle["prob_threshold"]
    target: str = bundle["target"]
    wer_thr: float = bundle["wer_threshold"]

    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise ValueError(f"Fehlende Feature-Spalten in df_features: {missing}")

    X = df_features[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)

    x_tensor = torch.from_numpy(X_scaled).to(dev)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    labels = (probs >= prob_thr).astype(int)

    out = pd.DataFrame()
    if "filename" in df_features.columns:
        out["filename"] = df_features["filename"]
    elif "file" in df_features.columns:
        out["filename"] = df_features["file"]

    out[f"{target}_thr_{wer_thr:.2f}_prob_good"] = probs
    out[f"{target}_thr_{wer_thr:.2f}_label_good"] = labels

    return out