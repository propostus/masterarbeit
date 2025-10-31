# scripts/train_multioutput_pytorch.py

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MultiOutputRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def evaluate_model(y_true, y_pred, label_names):
    results = []
    for i, label in enumerate(label_names):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        results.append({"target": label, "r2": r2, "rmse": rmse, "mae": mae})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Trainiere Multi-Output PyTorch Modell für WER Tiny/Base/Small")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weights", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Gewichte für [wer_tiny, wer_base, wer_small] im Loss")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== Multi-Output PyTorch Training auf {args.dataset} ===")

    df = pd.read_csv(args.dataset)
    non_numerical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numerical = [c for c in non_numerical if c not in ["filename", "snr"]]
    df = df.drop(columns=non_numerical, errors="ignore")

    target_cols = ["wer_tiny", "wer_base", "wer_small"]
    X = df.drop(columns=target_cols + ["filename", "snr"], errors="ignore")
    y = df[target_cols].values

    groups = df["filename"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
    y_train, y_val = y[train_idx], y[val_idx]

    overlap = set(df.loc[train_idx, "filename"]).intersection(df.loc[val_idx, "filename"])
    if overlap:
        print(f"Warnung: {len(overlap)} Dateien sind in beiden Splits!")
    else:
        print("Split-Check bestanden: Keine Überschneidung zwischen Train- und Validation-Files.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiOutputRegressor(input_dim=X_train.shape[1], output_dim=len(target_cols)).to(device)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.tensor(args.weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.float32)),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.float32)),
                            batch_size=args.batch_size, shuffle=False)

    print(f"Training startet ({args.epochs} Epochen, Batch Size={args.batch_size}, LR={args.lr})")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)

        train_loss /= len(train_loader.dataset)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                preds_val = []
                for Xb, _ in val_loader:
                    preds_val.append(model(Xb.to(device)).cpu().numpy())
                preds_val = np.vstack(preds_val)
            val_loss = mean_squared_error(y_val, preds_val)
            print(f"Epoche {epoch + 1}/{args.epochs} | Train Loss={train_loss:.4f} | Val MSE={val_loss:.4f}")

    model.eval()
    with torch.no_grad():
        preds = []
        for Xb, _ in val_loader:
            preds.append(model(Xb.to(device)).cpu().numpy())
        preds = np.vstack(preds)

    results_df = evaluate_model(y_val, preds, target_cols)
    print("\nErgebnisse (Validierung):")
    print(results_df)

    results_df.to_csv(os.path.join(args.out_dir, "pytorch_multioutput_results.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "pytorch_multioutput_model.pt"))
    print(f"\nModell und Ergebnisse gespeichert unter {args.out_dir}")


if __name__ == "__main__":
    main()