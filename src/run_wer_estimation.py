# src/run_wer_estimation.py

import os
import json
import argparse
import warnings

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
import joblib

from features import extract_sigmos_and_wavlm_features, resolve_device

# Warnungen von torchaudio und einigen deprecated Sachen unterdrücken
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


# -------------------------------------------------------------------
# Modelle
# -------------------------------------------------------------------
class MultiOutputMLP(nn.Module):
    """Regressions-MLP mit drei Ausgängen: wer_tiny, wer_base, wer_small"""

    def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmallMLP(nn.Module):
    """Binärer Klassifikator (eine Logit-Ausgabe)."""

    def __init__(self, input_dim: int, hidden_sizes: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -------------------------------------------------------------------
# Regression: Laden & Anwenden
# -------------------------------------------------------------------
def load_regression_model(device: torch.device):
    """
    Lädt das Multi-Output-MLP für die WER-Regression + Scaler + Featureliste.
    Erwartete Dateien in models/regression/:
      - mlp_regression_multiwer.pt
      - scaler.pkl
      - feature_cols.txt
      - mlp_regression_config_and_metrics.json
    """
    base_dir = "models/regression"
    model_path = os.path.join(base_dir, "mlp_regression_multiwer.pt")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    feat_path = os.path.join(base_dir, "feature_cols.txt")
    config_path = os.path.join(base_dir, "mlp_regression_config_and_metrics.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(scaler_path)
    if not os.path.exists(feat_path):
        raise FileNotFoundError(feat_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)

    # Feature-Spalten laden
    with open(feat_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # Konfiguration lesen (hidden_sizes, dropout)
    with open(config_path) as f:
        cfg = json.load(f)["best_config"]

    hidden_sizes = cfg["hidden_sizes"]
    dropout = cfg["dropout"]

    input_dim = len(feature_cols)
    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        output_dim=3,
        dropout=dropout,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    scaler: StandardScaler = joblib.load(scaler_path)

    return model, scaler, feature_cols


def predict_regression(df_features: pd.DataFrame, device: str = "auto") -> pd.DataFrame:
    """
    Wendet das Regressionsmodell auf bereits extrahierte Features an.
    Gibt DataFrame mit Spalten wer_tiny, wer_base, wer_small zurück.
    """
    dev = resolve_device(device)
    print(f"[Regression] Verwende Gerät: {dev}")

    model, scaler, feature_cols = load_regression_model(dev)

    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise RuntimeError(
            f"Feature-Spalten fehlen in den extrahierten Features: {missing[:10]} "
            f"(insgesamt {len(missing)})"
        )

    X = df_features[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        xb = torch.from_numpy(X_scaled).to(dev)
        preds = model(xb).cpu().numpy()

    df_pred = pd.DataFrame(
        preds,
        columns=["wer_tiny", "wer_base", "wer_small"],
    )
    return df_pred


def run_regression_on_folder(audio_dir: str, device: str, out_csv: str):
    print("[Regression] Extrahiere Features...")
    df_feat = extract_sigmos_and_wavlm_features(audio_dir=audio_dir, device=device)

    if df_feat.empty:
        print("[Regression] Keine Features extrahiert, breche ab.")
        return

    print("[Regression] WER-Regression anwenden...")
    df_pred = predict_regression(df_feat, device=device)

    base_cols = ["filename"]
    if "filepath" in df_feat.columns:
        base_cols.append("filepath")

    df_out = pd.concat(
        [df_feat[base_cols].reset_index(drop=True),
         df_pred.reset_index(drop=True)],
        axis=1
    )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[Regression] Fertig. Ergebnisse gespeichert unter: {out_csv}")


# -------------------------------------------------------------------
# Klassifikation: Laden & Anwenden
# -------------------------------------------------------------------

# Mapping der 9 Klassifikationsmodelle:
# (asr_model, wer_threshold) -> Thresh-String, Hidden-Sizes, Dropout, Prob-Threshold
BINARY_CONFIGS: Dict[Tuple[str, float], Dict] = {
    # tiny
    ("tiny", 0.05): {"tag": "005", "hidden_sizes": [512, 256], "dropout": 0.3, "prob_thr": 0.85},
    ("tiny", 0.10): {"tag": "010", "hidden_sizes": [512, 256], "dropout": 0.3, "prob_thr": 0.75},
    ("tiny", 0.20): {"tag": "020", "hidden_sizes": [512, 256], "dropout": 0.3, "prob_thr": 0.55},
    # base
    ("base", 0.05): {"tag": "005", "hidden_sizes": [512, 256], "dropout": 0.3, "prob_thr": 0.65},
    ("base", 0.10): {"tag": "010", "hidden_sizes": [512, 256], "dropout": 0.3, "prob_thr": 0.55},
    ("base", 0.20): {"tag": "020", "hidden_sizes": [512, 256, 128], "dropout": 0.3, "prob_thr": 0.30},
    # small
    ("small", 0.05): {"tag": "005", "hidden_sizes": [512, 256], "dropout": 0.3, "prob_thr": 0.35},
    ("small", 0.10): {"tag": "010", "hidden_sizes": [512, 256], "dropout": 0.2, "prob_thr": 0.25},
    ("small", 0.20): {"tag": "020", "hidden_sizes": [512, 256], "dropout": 0.3, "prob_thr": 0.10},
}


def load_binary_model(asr_model: str, wer_thr: float, input_dim: int, device: torch.device):
    """
    Lädt einen binären Klassifikator + Scaler + Prob-Threshold für
    ein bestimmtes Whisper-Modell und eine bestimmte WER-Schwelle.
    """
    key = (asr_model, wer_thr)
    if key not in BINARY_CONFIGS:
        raise ValueError(f"Keine Konfiguration für {key}")

    cfg = BINARY_CONFIGS[key]
    tag = cfg["tag"]

    base_dir = "models/classification"
    model_path = os.path.join(base_dir, f"mlp_binary_wer_{asr_model}_thr{tag}.pt")
    scaler_path = os.path.join(base_dir, f"mlp_binary_wer_{asr_model}_thr{tag}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(scaler_path)

    scaler: StandardScaler = joblib.load(scaler_path)

    model = SmallMLP(
        input_dim=input_dim,
        hidden_sizes=cfg["hidden_sizes"],
        dropout=cfg["dropout"],
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    prob_thr = cfg["prob_thr"]
    return model, scaler, prob_thr


def predict_classification(df_features: pd.DataFrame, device: str = "auto") -> pd.DataFrame:
    """
    Wendet alle binären Klassifikationsmodelle (tiny, base, small) für
    die drei WER-Schwellen 0.05, 0.10, 0.20 an.

    Ausgabe-Spalten (0/1):
      wer_tiny_under_5_percent
      wer_tiny_under_10_percent
      wer_tiny_under_20_percent
      wer_base_under_5_percent
      wer_base_under_10_percent
      wer_base_under_20_percent
      wer_small_under_5_percent
      wer_small_under_10_percent
      wer_small_under_20_percent
    """
    dev = resolve_device(device)
    print(f"[Classification] Verwende Gerät: {dev}")

    # Feature-Spalten wie im Regressionstraining verwenden
    feat_path = "models/regression/feature_cols.txt"
    if not os.path.exists(feat_path):
        raise FileNotFoundError(feat_path)

    with open(feat_path) as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise RuntimeError(
            f"Feature-Spalten fehlen in den extrahierten Features: {missing[:10]} "
            f"(insgesamt {len(missing)})"
        )

    X = df_features[feature_cols].values.astype(np.float32)
    input_dim = X.shape[1]

    base_cols = ["filename"]
    if "filepath" in df_features.columns:
        base_cols.append("filepath")
    df_out = df_features[base_cols].copy()

    # Alle 3 ASR-Modelle x 3 Schwellen
    asr_models = ["tiny", "base", "small"]
    thresholds = [0.05, 0.10, 0.20]

    for asr in asr_models:
        print(f"[Classification] WER-Klassifikation für Whisper-{asr} anwenden...")
        for thr in thresholds:
            print(f"[Classification]  Lade Modell für {asr}, WER <= {thr:.2f}")
            model, scaler, prob_thr = load_binary_model(asr, thr, input_dim, dev)

            X_scaled = scaler.transform(X)
            xb = torch.from_numpy(X_scaled).to(dev)

            with torch.no_grad():
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()

            y_hat = (probs >= prob_thr).astype(int)

            col_name = f"wer_{asr}_under_{int(thr * 100)}_percent"
            df_out[col_name] = y_hat

    return df_out


def run_classification_on_folder(audio_dir: str, device: str, out_csv: str):
    print("[Classification] Extrahiere Features...")
    df_feat = extract_sigmos_and_wavlm_features(audio_dir=audio_dir, device=device)

    if df_feat.empty:
        print("[Classification] Keine Features extrahiert, breche ab.")
        return

    df_pred = predict_classification(df_feat, device=device)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_pred.to_csv(out_csv, index=False)
    print(f"[Classification] Fertig. Ergebnisse gespeichert unter: {out_csv}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="WER-Schätzung (Regression & Klassifikation) auf Ordner von Audiodateien."
    )

    parser.add_argument("--audio_dir", type=str, required=True, help="Ordner mit Audiodateien")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["regression", "classification"],
        required=True,
        help="regression: kontinuierliche WER-Prognose; "
             "classification: 0/1-Gates für mehrere Schwellen und Whisper-Modelle",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu | cuda | mps | auto",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Pfad zur Ausgabe-CSV",
    )

    args = parser.parse_args()

    print(f"Audio-Ordner: {args.audio_dir}")
    print(f"Modus: {args.mode}")
    print(f"Gerät: {args.device}")
    print(f"Ausgabe: {args.out_csv}")

    if args.mode == "regression":
        run_regression_on_folder(
            audio_dir=args.audio_dir,
            device=args.device,
            out_csv=args.out_csv,
        )
    else:
        run_classification_on_folder(
            audio_dir=args.audio_dir,
            device=args.device,
            out_csv=args.out_csv,
        )


if __name__ == "__main__":
    main()