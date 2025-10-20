# scripts/train_wavlm_mlp.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


def train_mlp(X_train, y_train, X_val, y_val, input_dim, device, out_dir,
              hidden_dims=[256, 128], dropout=0.2, lr=1e-3, batch_size=64, epochs=100, patience=10):

    model = MLPRegressor(input_dim, hidden_dims, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False
    )

    best_r2 = -np.inf
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

        r2 = r2_score(val_targets, val_preds)
        mae = mean_absolute_error(val_targets, val_preds)
        rmse = mean_squared_error(val_targets, val_preds, squared=False)

        print(f"Epoch {epoch+1:03d}: loss={np.mean(train_losses):.4f}, val_r2={r2:.4f}, val_mae={mae:.4f}, val_rmse={rmse:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping.")
            break

    return best_r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--use_pca", action="store_true", help="Optionale PCA-Reduktion aktivieren")
    parser.add_argument("--pca_components", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n=== Training MLP on {os.path.basename(args.dataset_csv)} ===")

    df = pd.read_csv(args.dataset_csv)
    X = df.drop(columns=[args.target_col, "filename"], errors="ignore").select_dtypes(include=[np.number]).values
    y = df[args.target_col].values

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardisierung
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # PCA optional
    if args.use_pca:
        print(f"→ PCA auf {args.pca_components} Komponenten angewendet")
        pca = PCA(n_components=args.pca_components, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Verwende Gerät: {device}")

    best_r2 = train_mlp(
        X_train, y_train, X_val, y_val,
        input_dim=X_train.shape[1],
        device=device,
        out_dir=args.out_dir
    )

    print(f"\nBestes Validierungs-R²: {best_r2:.4f}")
    with open(os.path.join(args.out_dir, "result.txt"), "w") as f:
        f.write(f"Best R²: {best_r2:.4f}\n")


if __name__ == "__main__":
    main()