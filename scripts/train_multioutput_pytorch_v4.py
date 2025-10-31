# scripts/train_multioutput_pytorch_v4.py
# Multi-Output PyTorch-Modell zur WER-Vorhersage (tiny/base/small)
# Verbesserte Version mit:
#   - Gruppiertem Split (kein Leakage)
#   - SNR-Verteilungsprüfung
#   - Early Stopping
#   - Sauberes Speichern des vollständigen Modells (nicht nur state_dict)

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import Counter


# ------------------------------------------------------------
# 1. Multioutput-Netzwerk
# ------------------------------------------------------------
class MultiOutputRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MultiOutputRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# 2. Trainingsfunktion mit Early Stopping
# ------------------------------------------------------------
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, out_dir, epochs=100, patience=15):
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Speichere sowohl state_dict als auch komplettes Modell
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model_state.pt"))
            torch.save(model, os.path.join(out_dir, "best_model_full.pt"))
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping nach {epoch+1} Epochen (keine Verbesserung seit {patience} Epochen).")
            break

    return model


# ------------------------------------------------------------
# 3. Evaluation
# ------------------------------------------------------------
def evaluate_model(model, X, y, device, out_path=None, label="Validation"):
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X).to(device)).cpu().numpy()

    results = []
    for i, target in enumerate(["wer_tiny", "wer_base", "wer_small"]):
        r2 = r2_score(y[:, i], y_pred[:, i])
        rmse = mean_squared_error(y[:, i], y_pred[:, i], squared=False)
        mae = mean_absolute_error(y[:, i], y_pred[:, i])
        results.append({"target": target, "r2": r2, "rmse": rmse, "mae": mae})

    df = pd.DataFrame(results)
    print(f"\nErgebnisse ({label}):")
    print(df)
    if out_path:
        df.to_csv(out_path, index=False)
    return df


# ------------------------------------------------------------
# 4. SNR-Verteilungscheck
# ------------------------------------------------------------
def check_snr_distribution(train_df, val_df):
    print("\n=== SNR-Verteilung (Train vs. Validation) ===")
    for split_name, df in [("Train", train_df), ("Validation", val_df)]:
        counts = Counter(df["snr"])
        total = sum(counts.values())
        dist = {k: f"{(v/total)*100:.1f}%" for k, v in counts.items()}
        print(f"{split_name}: {dist}")


# ------------------------------------------------------------
# 5. Hauptlogik
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Multioutput PyTorch Regressor (WER tiny/base/small)")
    parser.add_argument("--dataset", required=True, help="Pfad zur Trainings-CSV")
    parser.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis")
    parser.add_argument("--epochs", type=int, default=100, help="Anzahl Trainings-Epochen")
    parser.add_argument("--use_scheduler", action="store_true", help="Nutze LR Scheduler")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== Multi-Output PyTorch Training auf {args.dataset} ===")

    # --------------------------------------------------------
    # Daten laden
    # --------------------------------------------------------
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    exclude_cols = ["filename", "snr", "wer_tiny", "wer_base", "wer_small", "reference", "hypothesis"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    # --------------------------------------------------------
    # Gruppierter Split (kein Leakage über SNR)
    # --------------------------------------------------------
    print("=== Erstelle gruppierten Split nach filename (alle SNR-Versionen gemeinsam) ===")
    groups = df["filename"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in gss.split(df, groups=groups):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

    overlap = set(train_df["filename"]) & set(val_df["filename"])
    if len(overlap) > 0:
        print(f"WARNUNG: {len(overlap)} Dateien erscheinen in beiden Splits!")
    else:
        print("Split-Check bestanden: Keine Überschneidung zwischen Train- und Validation-Files.")
    print(f"Trainingsset: {train_df.shape}, Validierungsset: {val_df.shape}")

    check_snr_distribution(train_df, val_df)

    # --------------------------------------------------------
    # Features & Targets
    # --------------------------------------------------------
    X_train = train_df[feature_cols].astype(np.float32).values
    y_train = train_df[["wer_tiny", "wer_base", "wer_small"]].astype(np.float32).values
    X_val = val_df[feature_cols].astype(np.float32).values
    y_val = val_df[["wer_tiny", "wer_base", "wer_small"]].astype(np.float32).values

    # --------------------------------------------------------
    # Torch Setup
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiOutputRegressor(input_dim=X_train.shape[1]).to(device)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=128, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
        if args.use_scheduler else None
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    model = train_model(model, train_loader, val_loader, optimizer, scheduler, device, args.out_dir, epochs=args.epochs)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model_state.pt"), map_location=device))
    val_results = evaluate_model(model, X_val, y_val, device, os.path.join(args.out_dir, "validation_metrics.csv"))


if __name__ == "__main__":
    main()