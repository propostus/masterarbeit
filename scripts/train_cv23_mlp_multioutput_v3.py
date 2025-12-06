# scripts/train_cv23_mlp_multioutput_v3.py
# ------------------------------------------------------
# Multi-Output-MLP für WER-Prediction (wer_tiny, wer_base, wer_small)
# Daten: CV23-balanced, nur SigMOS + WavLM (keine handcrafted-Features)
# Split: group-basiert nach client_id (Train / Val / Test)
# Skaliert Features mit StandardScaler (nur auf Train)
# Speichert:
#   - mlp_multioutput_v3_best.pt (state_dict)
#   - scaler_mlp_v3.pkl (StandardScaler)
#   - feature_cols_mlp_v3.txt
#   - config_mlp_v3.json (Hyperparameter)
#   - mlp_test_metrics_v3.csv (Metriken auf Test-Set)
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


def select_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA nicht verfügbar, verwende CPU.")
        return torch.device("cpu")
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS nicht verfügbar, verwende CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom == 0:
        return 0.0
    return (2 * cov) / denom


class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 3))  # 3 Outputs: wer_tiny, wer_base, wer_small
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_one_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=80,
    lr=1e-3,
    weight_decay=1e-5,
    patience=10,
    out_dir=".",
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n_train += xb.size(0)

        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                n_val += xb.size(0)
        val_loss /= max(n_val, 1)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early Stopping nach {epoch} Epochen (keine Verbesserung seit {patience} Epochen).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser(
        description="Train Multi-Output-MLP auf CV23-balanced (nur SigMOS+WavLM, v3)."
    )
    parser.add_argument("--dataset", required=True, help="Pfad zur CV23-balanced-CSV (SigMOS+WavLM+WER)")
    parser.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis für MLP v3")
    parser.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = select_device(args.device)
    print(f"Verwende Gerät: {device}")

    print(f"Lade Datensatz von: {args.dataset}")
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Datensatz: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    targets = ["wer_tiny", "wer_base", "wer_small"]

    exclude_cols = [
        "filename",
        "client_id",
        "age",
        "gender",
        "sentence",
        "wer_tiny",
        "wer_base",
        "wer_small",
        "reference",
        "hypothesis",
    ]

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    print(f"Anzahl Feature-Spalten: {len(feature_cols)}")

    before = len(df)
    df = df.dropna(subset=targets)
    after = len(df)
    if after < before:
        print(f"Droppe {before - after} Zeilen mit NaN in Targets. Verbleibend: {after}")

    if "client_id" in df.columns:
        groups = df["client_id"].astype(str)
    else:
        print("Warnung: Spalte 'client_id' fehlt, nutze 'filename' als Gruppe.")
        groups = df["filename"].astype(str)

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_state)
    idx_trainval, idx_test = next(gss_outer.split(df, groups=groups))
    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    groups_trainval = df_trainval["client_id"].astype(str) if "client_id" in df_trainval.columns else df_trainval["filename"].astype(str)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=args.random_state)
    idx_train, idx_val = next(gss_inner.split(df_trainval, groups=groups_trainval))

    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val = df_trainval.iloc[idx_val].reset_index(drop=True)

    print(f"Train: {len(df_train)}  |  Val: {len(df_val)}  |  Test: {len(df_test)}")
    print(f"Unique Speaker (Train/Val/Test): "
          f"{df_train['client_id'].nunique() if 'client_id' in df_train.columns else 'n/a'} / "
          f"{df_val['client_id'].nunique() if 'client_id' in df_val.columns else 'n/a'} / "
          f"{df_test['client_id'].nunique() if 'client_id' in df_test.columns else 'n/a'}")

    X_train = df_train[feature_cols].astype(np.float32).values
    X_val = df_val[feature_cols].astype(np.float32).values
    X_test = df_test[feature_cols].astype(np.float32).values

    y_train = df_train[targets].astype(np.float32).values
    y_val = df_val[targets].astype(np.float32).values
    y_test = df_test[targets].astype(np.float32).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]
    print(f"Input-Dimension: {input_dim}, Hidden Sizes: {args.hidden_sizes}")

    train_ds = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = MultiOutputMLP(input_dim=input_dim, hidden_sizes=args.hidden_sizes, dropout=args.dropout).to(device)

    model, best_val_loss = train_one_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=10,
        out_dir=args.out_dir,
    )

    model_path = os.path.join(args.out_dir, "mlp_multioutput_v3_best.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Bestes MLP-Modell gespeichert unter: {model_path}")

    import joblib
    scaler_path = os.path.join(args.out_dir, "scaler_mlp_v3.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler gespeichert unter: {scaler_path}")

    feature_path = os.path.join(args.out_dir, "feature_cols_mlp_v3.txt")
    with open(feature_path, "w") as f:
        for c in feature_cols:
            f.write(c + "\n")
    print(f"Feature-Liste gespeichert unter: {feature_path}")

    config = {
        "input_dim": input_dim,
        "hidden_sizes": args.hidden_sizes,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "random_state": args.random_state,
    }
    config_path = os.path.join(args.out_dir, "config_mlp_v3.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config gespeichert unter: {config_path}")

    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            ys.append(yb.numpy())

    preds = np.vstack(preds)
    ys = np.vstack(ys)

    results = []
    for i, target in enumerate(targets):
        y_true = ys[:, i]
        y_pred = preds[:, i]
        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        ccc = concordance_correlation_coefficient(y_true, y_pred)
        results.append(
            {
                "model": "MLP_multioutput_v3",
                "target": target,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "ccc": ccc,
            }
        )

    results_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "mlp_test_metrics_v3.csv")
    results_df.to_csv(out_csv, index=False)
    print("\n=== Test-Metriken (MLP v3, nur SigMOS+WavLM) ===")
    print(results_df)
    print(f"\nMetriken gespeichert unter: {out_csv}")


if __name__ == "__main__":
    main()