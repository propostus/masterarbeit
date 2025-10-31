# scripts/train_multioutput_pytorch_v2.py

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------
def r2_loss(y_pred, y_true):
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return 1 - torch.mean(r2)


def evaluate_model(model, loader, criterion, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            y_true.append(yb.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    return r2, rmse, mae


# ---------------------------------------------
# Modellarchitektur
# ---------------------------------------------
class MultiOutputNet(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------
# Hauptfunktion
# ---------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--use_snr", action="store_true", help="SNR als numerisches Feature verwenden")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== Multi-Output PyTorch Training auf {args.dataset} ===")
    df = pd.read_csv(args.dataset)

    # Nicht-numerische Spalten entfernen
    drop_cols = ["filename", "source_embed", "source_hand", "rt60_method"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Optional: SNR als numerisches Feature hinzufügen
    if args.use_snr and "snr" in df.columns:
        snr_map = {"clean": 30, "20": 20, "10": 10, "0": 0}
        df["snr_value"] = df["snr"].astype(str).map(snr_map)
    df = df.drop(columns=["snr"], errors="ignore")

    # Features und Targets
    feature_cols = [c for c in df.columns if not c.startswith("wer_")]
    target_cols = [c for c in df.columns if c.startswith("wer_")]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    # Gruppierung nach Basis-Filename (damit clean/noisy zusammenbleiben)
    groups = df["filename"].apply(lambda x: os.path.basename(x).split(".")[0]) if "filename" in df.columns else np.arange(len(df))
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Trainingsset: {X_train.shape}, Validierungsset: {X_val.shape}")

    # Torch-Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiOutputNet(input_dim=X.shape[1], output_dim=y.shape[1]).to(device)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=256)

    criterion = r2_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # Training mit Early Stopping
    best_r2 = -np.inf
    best_epoch = 0
    patience = 25
    epochs = 200

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validierung
        r2, rmse, mae = evaluate_model(model, val_loader, criterion, device)
        mean_r2 = np.mean(r2)
        scheduler.step(-mean_r2.item() if isinstance(mean_r2, torch.Tensor) else -mean_r2)

        print(f"Epoche {epoch+1:03d}: Loss={epoch_loss/len(train_loader):.4f}, R²={mean_r2:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
        elif epoch - best_epoch > patience:
            print("Early stopping aktiviert.")
            break

    print(f"Bestes Modell: Epoche {best_epoch+1}, R²={best_r2:.4f}")

    # Finalauswertung mit bestem Modell
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pt")))
    r2, rmse, mae = evaluate_model(model, val_loader, criterion, device)

    results = pd.DataFrame({
        "target": target_cols,
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    })
    results.to_csv(os.path.join(args.out_dir, "validation_results.csv"), index=False)
    print("\nErgebnisse (Validierung):")
    print(results)


if __name__ == "__main__":
    main()