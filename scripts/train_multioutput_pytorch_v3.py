# scripts/train_multioutput_pytorch_v3.py
# Multi-Output PyTorch-Modell zur WER-Vorhersage (tiny/base/small)
# Mit gruppiertem Split nach filename, um Leakage über SNR-Stufen zu vermeiden

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

# ------------------------------------------------------------
# 1. Multioutput-Netzwerk
# ------------------------------------------------------------
class MultiOutputRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MultiOutputRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # WER tiny, base, small
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# 2. Trainingsfunktion
# ------------------------------------------------------------
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, out_dir, epochs=50):
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

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

        # Bester Checkpoint speichern
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    return model


# ------------------------------------------------------------
# 3. Hauptlogik
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Multioutput PyTorch Regressor (WER tiny/base/small)")
    parser.add_argument("--dataset", required=True, help="Pfad zur CSV-Datei")
    parser.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis")
    parser.add_argument("--epochs", type=int, default=50, help="Anzahl Trainings-Epochen")
    parser.add_argument("--use_scheduler", action="store_true", help="Nutze LR Scheduler")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== Multi-Output PyTorch Training auf {args.dataset} ===")

    # --------------------------------------------------------
    # Daten laden
    # --------------------------------------------------------
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Nicht-numerische und Zielspalten ausschließen
    exclude_cols = ["filename", "snr", "wer_tiny", "wer_base", "wer_small"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    # --------------------------------------------------------
    # Gruppierter Split nach filename (keine Leakage über SNR)
    # --------------------------------------------------------
    print("=== Erstelle gruppierten Split nach filename (alle SNR-Versionen gemeinsam) ===")
    groups = df["filename"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in gss.split(df, groups=groups):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

    # Leakage-Prüfung
    overlap = set(train_df["filename"]) & set(val_df["filename"])
    if len(overlap) > 0:
        print(f"⚠️  WARNUNG: {len(overlap)} Dateien erscheinen in beiden Splits!")
        print("Beispielhafte Überschneidungen:", list(overlap)[:5])
    else:
        print("✅ Split-Check bestanden: Keine Überschneidung zwischen Train- und Validation-Files.")
    print(f"Trainingsset: {train_df.shape}, Validierungsset: {val_df.shape}")

    # --------------------------------------------------------
    # Features & Targets
    # --------------------------------------------------------
    X_train = train_df[feature_cols].astype(np.float32).values
    y_train = train_df[["wer_tiny", "wer_base", "wer_small"]].astype(np.float32).values
    X_val = val_df[feature_cols].astype(np.float32).values
    y_val = val_df[["wer_tiny", "wer_base", "wer_small"]].astype(np.float32).values

    # --------------------------------------------------------
    # Torch-Datasets
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiOutputRegressor(input_dim=X_train.shape[1]).to(device)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=128, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10) if args.use_scheduler else None

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    model = train_model(model, train_loader, val_loader, optimizer, scheduler, device, args.out_dir, epochs=args.epochs)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pt"), map_location=device))
    model.eval()

    with torch.no_grad():
        y_pred = model(torch.tensor(X_val).to(device)).cpu().numpy()

    results = []
    for i, target in enumerate(["wer_tiny", "wer_base", "wer_small"]):
        r2 = r2_score(y_val[:, i], y_pred[:, i])
        rmse = mean_squared_error(y_val[:, i], y_pred[:, i], squared=False)
        mae = mean_absolute_error(y_val[:, i], y_pred[:, i])
        results.append({"target": target, "r2": r2, "rmse": rmse, "mae": mae})

    results_df = pd.DataFrame(results)
    print("\nErgebnisse (Validierung):")
    print(results_df)
    results_df.to_csv(os.path.join(args.out_dir, "validation_metrics.csv"), index=False)


if __name__ == "__main__":
    main()