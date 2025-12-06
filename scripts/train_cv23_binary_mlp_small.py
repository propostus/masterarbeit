# scripts/train_cv23_binary_mlp_small.py
import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import joblib
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def get_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_str == "mps":
        return torch.device("mps")
    if device_str == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def make_split(df: pd.DataFrame, groups_col: str, test_size: float, val_size: float, random_state: int):
    """GroupShuffleSplit: Train / Val / Test nach client_id."""
    groups = df[groups_col].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss.split(df, groups=groups))

    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    groups_tv = train_val[groups_col].astype(str)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=random_state)
    train_idx, val_idx = next(gss2.split(train_val, groups=groups_tv))

    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    return train, val, test


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # binary logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_mlp(
    X_train, y_train, X_val, y_val, device,
    max_epochs=20, batch_size=512, lr=1e-3, pos_weight=None
):
    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)),
        torch.from_numpy(y_val.astype(np.float32)),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = SmallMLP(input_dim=X_train.shape[1]).to(device)

    if pos_weight is not None:
        # muss float32 auf dem Zielgerät sein (MPS-kompatibel)
        pw = torch.tensor(float(pos_weight), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_auc = -np.inf
    best_state = None
    patience = 5
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        all_logits = []
        all_y = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                all_logits.append(logits.cpu().numpy())
                all_y.append(yb.numpy())

        all_logits = np.concatenate(all_logits)
        all_y = np.concatenate(all_y)
        probs = 1 / (1 + np.exp(-all_logits))
        try:
            val_auc = roc_auc_score(all_y, probs)
        except ValueError:
            val_auc = 0.5

        print(f"Epoch {epoch:02d}: train_loss={avg_train_loss:.4f}, val_roc_auc={val_auc:.4f}")

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early Stopping nach {epoch} Epochen (best val AUC={best_val_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_auc


def evaluate_binary(model, X, y, device):
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_logits.append(logits.cpu().numpy())
            all_y.append(yb.numpy())
    all_logits = np.concatenate(all_logits)
    all_y = np.concatenate(all_y)
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(all_y, preds)
    f1 = f1_score(all_y, preds, zero_division=0)
    try:
        roc = roc_auc_score(all_y, probs)
    except ValueError:
        roc = np.nan
    try:
        pr = average_precision_score(all_y, probs)
    except ValueError:
        pr = np.nan

    return {
        "acc": acc,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr,
    }


# ------------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kleine MLP-Binary-Classifier für wer_tiny/wer_base/wer_small (cv23).")
    parser.add_argument("--dataset", required=True, help="Pfad zu merged_sigmos_wavlm_cv23_balanced_multiwer.csv")
    parser.add_argument("--out_dir", required=True, help="Output-Verzeichnis")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Grenze: WER <= threshold -> Klasse 1 (Standard: 0.0 = perfekt)")
    parser.add_argument("--device", type=str, default="auto", help="cpu|cuda|mps|auto")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.device)
    print(f"Verwende Gerät: {device}")

    df = pd.read_csv(args.dataset)
    print(f"Shape Datensatz: {df.shape}")

    # Sicherstellen, dass client_id existiert
    if "client_id" not in df.columns:
        raise RuntimeError("Spalte 'client_id' wird für Speaker-Split benötigt.")

    # Feature-Spalten: nur numerische, ohne Targets
    target_cols = ["wer_tiny", "wer_base", "wer_small"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in target_cols]
    print(f"Anzahl Feature-Spalten: {len(feature_cols)}")

    # Feature-Array vorbereiten
    X_all = df[feature_cols].values.astype(np.float32)

    results = []

    for target in target_cols:
        print("\n" + "=" * 70)
        print(f"Zielvariable: {target} (WER <= {args.threshold:.2f} als Klasse 1)")
        print("=" * 70)

        wer_values = df[target].values.astype(np.float32)
        y_all = (wer_values <= args.threshold).astype(int)

        # Split nach client_id
        df_tmp = df.copy()
        df_tmp["y"] = y_all
        train_df, val_df, test_df = make_split(
            df_tmp, groups_col="client_id",
            test_size=0.2, val_size=0.1, random_state=args.random_state,
        )

        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        # Features passend extrahieren
        X_train = train_df[feature_cols].values.astype(np.float32)
        X_val = val_df[feature_cols].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_train = train_df["y"].values.astype(int)
        y_val = val_df["y"].values.astype(int)
        y_test = test_df["y"].values.astype(int)

        print(f"Label-Verteilung (Train) für {target}: {Counter(y_train)}")

        # StandardScaler
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        # class imbalance -> pos_weight
        counts = Counter(y_train)
        neg_count = counts.get(0, 1)
        pos_count = counts.get(1, 1)
        pos_weight = neg_count / max(pos_count, 1)
        print(f"pos_weight für {target}: {pos_weight:.2f}")

        # MLP trainieren
        model, best_val_auc = train_one_mlp(
            X_train_sc, y_train, X_val_sc, y_val,
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=5e-4,
            pos_weight=pos_weight,
        )

        print(f"Beste Val-ROC-AUC für {target}: {best_val_auc:.4f}")

        # Testevaluation
        test_metrics = evaluate_binary(model, X_test_sc, y_test, device)
        print(f"Test-Metriken für {target}: "
              f"acc={test_metrics['acc']:.3f}, "
              f"f1={test_metrics['f1']:.3f}, "
              f"roc_auc={test_metrics['roc_auc']:.3f}, "
              f"pr_auc={test_metrics['pr_auc']:.3f}")

        # speichern
        model_path = os.path.join(args.out_dir, f"mlp_small_{target}.pt")
        scaler_path = os.path.join(args.out_dir, f"scaler_mlp_small_{target}.pkl")
        torch.save(model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)

        results.append({
            "model": "MLP_small",
            "target": target,
            "threshold": args.threshold,
            "val_roc_auc": best_val_auc,
            "test_acc": test_metrics["acc"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
        })

    # Feature-Liste einmal speichern
    with open(os.path.join(args.out_dir, "feature_cols.txt"), "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "binary_mlp_small_metrics.csv")
    res_df.to_csv(out_csv, index=False)
    print("\n=== Fertig. Ergebnisse ===")
    print(res_df)
    print(f"\nMetriken gespeichert unter: {out_csv}")


if __name__ == "__main__":
    main()