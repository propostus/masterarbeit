# scripts/train_cv23_mlp_multioutput.py
# ------------------------------------------------------
# Multi-Output MLP für WER-Prediction (wer_tiny, wer_base, wer_small)
# auf CV23-balanced + SigMOS + WavLM + Handcrafted Features
#
# - Gruppierter Split nach client_id (kein Speaker-Leakage)
# - StandardScaler nur auf Trainingsdaten
# - Multi-Output-Regressor (3 Targets)
# - Early Stopping + ReduceLROnPlateau (ohne verbose)
# - Speichert: best_model.pt, scaler.pkl, feature_cols.txt, metrics.csv
# ------------------------------------------------------

import os
import json
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from joblib import dump


# ------------------------------------------------------
# Hilfsfunktionen / Metriken
# ------------------------------------------------------

def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance Correlation Coefficient (CCC) für 1D-Arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)
    return ccc


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            print("Verwende Gerät: cuda")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("Verwende Gerät: mps")
            return torch.device("mps")
        print("Verwende Gerät: cpu")
        return torch.device("cpu")
    else:
        dev = torch.device(device_arg)
        print(f"Verwende Gerät: {dev}")
        return dev


# ------------------------------------------------------
# MLP-Modell
# ------------------------------------------------------

class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes, dropout: float, output_dim: int = 3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------
# Training / Evaluation
# ------------------------------------------------------

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float()
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).float()
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / max(n_samples, 1)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            batch_size = X_batch.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

    return running_loss / max(n_samples, 1)


def evaluate_full(model, X, y_true, device):
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(device)).cpu().numpy()

    metrics = []
    targets = ["wer_tiny", "wer_base", "wer_small"]
    for i, name in enumerate(targets):
        yt = y_true[:, i]
        yp = preds[:, i]
        r2 = r2_score(yt, yp)
        rmse = mean_squared_error(yt, yp, squared=False)
        mae = mean_absolute_error(yt, yp)
        ccc = concordance_correlation_coefficient(yt, yp)
        metrics.append(
            {"target": name, "r2": r2, "rmse": rmse, "mae": mae, "ccc": ccc}
        )
    return metrics


# ------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Output MLP auf CV23-balanced (SigMOS + WavLM + Handcrafted).")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Pfad zu merged_sigmos_wavlm_cv23_balanced_multiwer.csv")
    parser.add_argument("--extra_features_csv", type=str, required=True,
                        help="Pfad zu handcrafted_audio_features_cv23_balanced.csv")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Ausgabeverzeichnis für Modell, Scaler, Metriken")
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu | cuda | mps | auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=10)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = select_device(args.device)

    # --------------------------------------------------
    # Daten laden und mergen
    # --------------------------------------------------
    print(f"Lade Datensatz von: {args.dataset}")
    df = pd.read_csv(args.dataset, low_memory=False)

    print(f"Lade zusätzliche Features von: {args.extra_features_csv}")
    df_extra = pd.read_csv(args.extra_features_csv, low_memory=False)

    # Normalisiere Join-Key
    for col in ["filename", "file"]:
        if col in df.columns:
            df["filename"] = df[col].astype(str)
            break
    if "filename" not in df.columns:
        raise RuntimeError("Im Haupt-Dataset wurde keine Spalte 'filename' oder 'file' gefunden.")

    for col in ["filename", "file"]:
        if col in df_extra.columns:
            df_extra["filename"] = df_extra[col].astype(str)
            break
    if "filename" not in df_extra.columns:
        raise RuntimeError("Im Extra-Feature-Dataset wurde keine Spalte 'filename' oder 'file' gefunden.")

    df = df.merge(df_extra, on="filename", how="left", suffixes=("", "_extra"))
    print(f"Nach Merge mit Extra-Features: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Targets
    target_cols = ["wer_tiny", "wer_base", "wer_small"]
    df = df.dropna(subset=target_cols)
    print(f"Nach Drop von NaN-Targets: {df.shape[0]} Zeilen")

    # --------------------------------------------------
    # Feature-Auswahl
    # --------------------------------------------------
    exclude_cols = set([
        "filename", "client_id", "age", "gender", "sentence",
        "wer_tiny", "wer_base", "wer_small",
        "reference", "hypothesis"
    ])

    numeric_cols = [c for c in df.columns
                    if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    feature_cols = numeric_cols
    print(f"Anzahl Feature-Spalten: {len(feature_cols)}")

    X_all = df[feature_cols].astype(np.float32).fillna(0.0).values
    y_all = df[target_cols].astype(np.float32).values

    # --------------------------------------------------
    # Gruppierter Split nach client_id
    # --------------------------------------------------
    if "client_id" not in df.columns:
        raise RuntimeError("Spalte 'client_id' fehlt im Datensatz, gruppierter Speaker-Split nicht möglich.")

    groups = df["client_id"].astype(str).values

    # Erst Train+Val vs. Test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainval_idx, test_idx = next(gss1.split(X_all, y_all, groups=groups))
    X_trainval, X_test = X_all[trainval_idx], X_all[test_idx]
    y_trainval, y_test = y_all[trainval_idx], y_all[test_idx]
    groups_trainval = groups[trainval_idx]

    # Innerhalb Train+Val -> Train vs. Val
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups=groups_trainval))

    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    print(f"Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"Unique Speaker (Train/Val/Test): "
          f"{len(np.unique(groups_trainval[train_idx]))} / "
          f"{len(np.unique(groups_trainval[val_idx]))} / "
          f"{len(np.unique(groups[test_idx]))}")

    # --------------------------------------------------
    # Skalierung
    # --------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Scaler und Feature-Liste speichern
    dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))
    with open(os.path.join(args.out_dir, "feature_cols.txt"), "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    # --------------------------------------------------
    # DataLoader + Modell
    # --------------------------------------------------
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
        batch_size=args.batch_size
    )

    input_dim = X_train_scaled.shape[1]
    print(f"Input-Dimension: {input_dim}, Hidden Sizes: {args.hidden_sizes}")

    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        output_dim=3
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # --------------------------------------------------
    # Trainingsloop mit Early Stopping
    # --------------------------------------------------
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early Stopping nach {epoch} Epochen (Best Val Loss={best_val_loss:.6f}).")
            break

    # Bestes Modell laden und speichern
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))

    # --------------------------------------------------
    # Evaluation auf Val + Test
    # --------------------------------------------------
    val_metrics = evaluate_full(model, X_val_scaled, y_val, device)
    test_metrics = evaluate_full(model, X_test_scaled, y_test, device)

    val_df = pd.DataFrame(val_metrics)
    test_df = pd.DataFrame(test_metrics)

    print("\nErgebnisse (Validation):")
    print(val_df)
    print("\nErgebnisse (Test):")
    print(test_df)

    val_df.to_csv(os.path.join(args.out_dir, "mlp_val_metrics.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "mlp_test_metrics.csv"), index=False)

    # Konfiguration speichern
    cfg_path = os.path.join(args.out_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nKonfiguration gespeichert unter: {cfg_path}")


if __name__ == "__main__":
    main()