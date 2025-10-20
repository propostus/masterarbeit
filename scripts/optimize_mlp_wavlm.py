# scripts/optimize_mlp_wavlm.py

import os
import argparse
import json
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout=0.0, activation="ReLU"):
        super().__init__()
        layers = []
        act_fn = getattr(nn, activation)()
        prev_dim = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev_dim, h), act_fn, nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, max_epochs=50, patience=5):
    best_r2 = -np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        val_r2 = r2_score(y_true, y_pred)

        print(f"Epoch {epoch:03d}: train_loss={train_loss/len(train_loader.dataset):.4f}, val_r2={val_r2:.4f}")

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    return model, best_r2


def objective(trial, X_train, X_val, y_train, y_val, device):
    hidden_layers = trial.suggest_int("hidden_layers", 1, 4)
    hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 3e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    activation = trial.suggest_categorical("activation", ["ReLU", "GELU"])

    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden_sizes=[hidden_size] * hidden_layers,
        dropout=dropout,
        activation=activation
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    _, best_r2 = train_model(model, train_loader, val_loader, optimizer, criterion, device)
    return best_r2


def optimize_mlp(dataset_csv, target_col, out_dir, use_pca=False, n_trials=100):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(dataset_csv)
    X = df.drop(columns=[target_col, "filename"], errors="ignore").select_dtypes(include=[np.number]).values
    y = df[target_col].values

    if use_pca:
        print("→ PCA auf 256 Komponenten angewendet")
        X = PCA(n_components=256, random_state=42).fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Verwende Gerät: {device}")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val, device), n_trials=n_trials)

    print(f"\nBest Trial R²: {study.best_value:.4f}")
    print("Beste Parameter:")
    print(study.best_params)

    best_params_path = os.path.join(out_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    results_path = os.path.join(out_dir, "optuna_results.csv")
    pd.DataFrame(study.trials_dataframe()).to_csv(results_path, index=False)

    print(f"\nErgebnisse gespeichert unter:\n{results_path}\n{best_params_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()

    optimize_mlp(args.dataset_csv, args.target_col, args.out_dir, args.use_pca, args.n_trials)