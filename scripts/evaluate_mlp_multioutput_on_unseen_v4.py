# scripts/evaluate_mlp_multioutput_on_unseen_v4.py
# ------------------------------------------------------
# Evaluiert das Multi-Output-MLP v4 auf dem UNSEEN-Datensatz.
# - Verwendet feature_cols.txt (vom Training)
# - Versucht scaler.pkl zu laden
# - Wenn scaler.pkl defekt/fehlt und --train_csv angegeben ist:
#       -> rekonstruiert den StandardScaler aus dem Trainingsdatensatz
# ------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ------------------------------------------------------
# CCC
# ------------------------------------------------------
def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)
    return ccc


# ------------------------------------------------------
# Multi-Output-MLP (v4-Architektur)
# ------------------------------------------------------
class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim=3, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------
# Gerät bestimmen
# ------------------------------------------------------
def get_device(arg_device: str) -> torch.device:
    if arg_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(arg_device)


# ------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluiert Multi-Output-MLP v4 auf UNSEEN-Datensatz."
    )
    parser.add_argument(
        "--unseen_csv",
        required=True,
        help="Pfad zur UNSEEN-CSV (merged_sigmos_wavlm_unseen.csv)",
    )
    parser.add_argument(
        "--models_dir",
        required=True,
        help="Ordner mit mlp_multioutput_best.pt (oder best_model.pt) und feature_cols.txt",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Pfad zur Ausgabe-CSV mit Metriken",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu | cuda | mps | auto",
    )
    parser.add_argument(
        "--train_csv",
        default=None,
        help="Optional: Pfad zur Trainings-CSV (z. B. merged_sigmos_wavlm_cv23_balanced_multiwer.csv), "
             "wird genutzt, um den Scaler zu rekonstruieren, falls scaler.pkl defekt ist.",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Verwende Gerät: {device}")

    # -----------------------------
    # 1. Unseen laden
    # -----------------------------
    print(f"Lade UNSEEN-Daten von: {args.unseen_csv}")
    df_unseen = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"Unseen-Shape: {df_unseen.shape}")

    target_cols = ["wer_tiny", "wer_base", "wer_small"]
    for col in target_cols:
        if col not in df_unseen.columns:
            raise ValueError(f"Spalte '{col}' nicht in UNSEEN-CSV gefunden.")

    # -----------------------------
    # 2. feature_cols & Scaler
    # -----------------------------
    feature_cols_path = os.path.join(args.models_dir, "feature_cols.txt")
    scaler_path = os.path.join(args.models_dir, "scaler.pkl")

    if not os.path.exists(feature_cols_path):
        raise FileNotFoundError(f"feature_cols.txt nicht gefunden unter: {feature_cols_path}")

    with open(feature_cols_path, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    print(f"Anzahl Feature-Spalten (laut Training): {len(feature_cols)}")

    # Prüfen, ob alle Features im UNSEEN vorhanden sind
    missing = [c for c in feature_cols if c not in df_unseen.columns]
    if missing:
        raise ValueError(f"Folgende Features fehlen in UNSEEN-CSV: {missing[:20]} ...")

    scaler = None
    if os.path.exists(scaler_path):
        try:
            print(f"Versuche scaler.pkl von: {scaler_path} zu laden ...")
            scaler = joblib.load(scaler_path)
            print("Scaler erfolgreich geladen.")
        except Exception as e:
            print(f"Warnung: Konnte scaler.pkl nicht laden: {e}")
            scaler = None
    else:
        print(f"Hinweis: scaler.pkl existiert nicht unter: {scaler_path}")

    # Wenn kein gültiger Scaler: optional aus Trainings-CSV rekonstruieren
    if scaler is None:
        if args.train_csv is None:
            raise RuntimeError(
                "Kein gültiger Scaler vorhanden und --train_csv wurde nicht angegeben. "
                "Bitte entweder scaler.pkl reparieren oder --train_csv setzen."
            )
        print(f"\nRekonstruiere StandardScaler aus Training-CSV: {args.train_csv}")
        df_train = pd.read_csv(args.train_csv, low_memory=False)
        # sicherstellen, dass die gleichen Features existieren
        missing_train = [c for c in feature_cols if c not in df_train.columns]
        if missing_train:
            raise ValueError(
                f"Folgende Features fehlen in TRAIN-CSV: {missing_train[:20]} ..."
            )
        X_train = df_train[feature_cols].values.astype(np.float32)
        scaler = StandardScaler()
        scaler.fit(X_train)
        print("Scaler aus Trainingsdaten rekonstruiert.")

    # -----------------------------
    # 3. X_unseen / y_unseen
    # -----------------------------
    X_unseen = df_unseen[feature_cols].values.astype(np.float32)
    y_unseen = df_unseen[target_cols].values.astype(np.float32)

    X_unseen_scaled = scaler.transform(X_unseen)
    X_tensor = torch.from_numpy(X_unseen_scaled).to(device)

    # -----------------------------
    # 4. Modell aufbauen & laden
    # -----------------------------
    hidden_sizes = [512, 256, 128]
    dropout = 0.3
    input_dim = X_tensor.shape[1]
    output_dim = 3

    print(
        f"Baue Modell mit input_dim={input_dim}, hidden_sizes={hidden_sizes}, "
        f"dropout={dropout}, output_dim={output_dim}"
    )

    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)

    # Modell-Datei finden
    cand_paths = [
        os.path.join(args.models_dir, "mlp_multioutput_best.pt"),
        os.path.join(args.models_dir, "best_model.pt"),
    ]
    model_path = None
    for p in cand_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        raise FileNotFoundError(
            f"Keine Modell-Datei gefunden unter {args.models_dir}. "
            f"Erwartet z. B. mlp_multioutput_best.pt oder best_model.pt."
        )

    print(f"Lade Modellgewichte von: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # -----------------------------
    # 5. Vorhersage
    # -----------------------------
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    # -----------------------------
    # 6. Metriken
    # -----------------------------
    rows = []
    for i, target in enumerate(target_cols):
        yt = y_unseen[:, i]
        yp = y_pred[:, i]

        r2 = r2_score(yt, yp)
        rmse = mean_squared_error(yt, yp, squared=False)
        mae = mean_absolute_error(yt, yp)
        ccc = concordance_correlation_coefficient(yt, yp)

        rows.append(
            {
                "model": "MLP_multioutput_v4",
                "target": target,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "ccc": ccc,
            }
        )

    r2_mean = np.mean([r["r2"] for r in rows])
    rmse_mean = np.mean([r["rmse"] for r in rows])
    mae_mean = np.mean([r["mae"] for r in rows])
    ccc_mean = np.mean([r["ccc"] for r in rows])

    rows.append(
        {
            "model": "MLP_multioutput_v4",
            "target": "mean_over_targets",
            "r2": r2_mean,
            "rmse": rmse_mean,
            "mae": mae_mean,
            "ccc": ccc_mean,
        }
    )

    results_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)

    print("\n=== Unseen-Evaluation (MLP v4) ===")
    print(results_df)
    print(f"\nMetriken gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()