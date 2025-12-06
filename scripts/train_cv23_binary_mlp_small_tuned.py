# scripts/train_cv23_binary_mlp_small_tuned.py
#
# Trainiert ein kleines MLP für die Binärklassifikation
# (WER <= 0.0 vs. > 0.0) für wer_tiny / wer_base / wer_small
# auf CV23 und macht Threshold-Tuning mit Fokus auf hoher Precision.
#
# Ausgaben:
#   - CSV mit Metriken pro Target und Threshold
#   - konsolenfreundliche Zusammenfassung

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Kleines MLP-Modell
# -----------------------------
class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -----------------------------
# Hilfsfunktionen
# -----------------------------
def build_features_and_groups(df: pd.DataFrame):
    # nicht-numerische / meta-Spalten entfernen
    drop_cols = [
        "filename",
        "client_id",
        "sentence",
        "age",
        "gender",
        "wer_tiny",
        "wer_base",
        "wer_small",
    ]
    feature_cols = [
        c for c in df.columns
        if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    X = df[feature_cols].values.astype(np.float32)
    groups = df["client_id"].astype(str).values
    return X, feature_cols, groups


def make_train_val_test_split(X, y, groups, test_size=0.2, val_size=0.15, random_state=42):
    # Erst Train+Val vs Test
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss.split(X, y, groups))

    X_trainval, y_trainval, groups_trainval = (
        X[trainval_idx],
        y[trainval_idx],
        groups[trainval_idx],
    )
    X_test, y_test = X[test_idx], y[test_idx]

    # Dann Train vs Val innerhalb Train+Val
    val_rel = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=random_state)
    train_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups_trainval))

    X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
    X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    device,
    max_epochs=20,
    batch_size=512,
    lr=1e-3,
    weight_decay=0.0,
    pos_weight=None,
):
    model = SmallMLP(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.float32))
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"  Epoch {epoch+1:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping.")
                break

    model.load_state_dict(best_state)
    return model


def eval_thresholds(y_true, probas, thresholds, min_recall=0.1):
    rows = []
    for t in thresholds:
        y_pred = (probas >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        rows.append(
            {
                "threshold": t,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "acc": acc,
            }
        )

    df = pd.DataFrame(rows)

    # Beste Schwelle nach Precision, aber mit Mindest-Recall
    df_valid = df[df["recall"] >= min_recall]
    if df_valid.empty:
        best_row = df.loc[df["precision"].idxmax()]
    else:
        best_row = df_valid.loc[df_valid["precision"].idxmax()]

    return df, best_row


# -----------------------------
# Hauptfunktion
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Kleines MLP für Binärklassifikation (WER==0 vs >0) mit Threshold-Tuning"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Pfad zur CV23-Trainings-CSV (merged_sigmos_wavlm_cv23_balanced_multiwer.csv)",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Pfad zur Output-CSV mit Threshold-Metriken",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="cpu, cuda, mps oder auto",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20,
        help="Maximale Trainingsepochen fürs kleine MLP",
    )
    args = parser.parse_args()

    # Device wählen
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

    # Daten laden
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Shape Datensatz: {df.shape}")

    X_all, feature_cols, groups = build_features_and_groups(df)
    print(f"Anzahl Feature-Spalten: {X_all.shape[1]}")

    results_all = []

    targets = ["wer_tiny", "wer_base", "wer_small"]

    for target in targets:
        print("\n" + "=" * 70)
        print(f"Zielvariable: {target} (WER == 0.0 als Klasse 1)")
        print("=" * 70)

        y_cont = df[target].values.astype(np.float32)
        y_bin = (y_cont <= 0.0).astype(int)

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        ) = make_train_val_test_split(X_all, y_bin, groups)

        print(f"Label-Verteilung (Train) für {target}: {Counter(y_train)}")

        # Skalierung
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        input_dim = X_train_sc.shape[1]

        # Class-Imbalance-Gewicht
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        pos_weight_val = float(neg_count / max(pos_count, 1))
        pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32, device=device)

        print("Trainiere MLP ...")
        model = train_mlp(
            X_train_sc,
            y_train,
            X_val_sc,
            y_val,
            input_dim=input_dim,
            device=device,
            max_epochs=args.max_epochs,
            batch_size=512,
            lr=1e-3,
            weight_decay=1e-4,
            pos_weight=pos_weight,
        )

        # Test-Probabilities
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_test_sc).to(device))
            probas = torch.sigmoid(logits).cpu().numpy()

        # Schwellen-Grid: im oberen Bereich feiner
        thresholds = np.concatenate(
            [
                np.linspace(0.1, 0.5, 9, endpoint=True),
                np.linspace(0.5, 0.95, 10, endpoint=True),
            ]
        )

        df_thr, best_row = eval_thresholds(
            y_test, probas, thresholds, min_recall=0.1
        )

        # ROC/PR-AUC zur Einordnung
        roc = roc_auc_score(y_test, probas)
        pr = average_precision_score(y_test, probas)

        print(f"\nBeste Schwelle für {target} (nach Precision, recall>=0.1):")
        print(best_row)
        print(f"ROC-AUC={roc:.3f}, PR-AUC={pr:.3f}")

        df_thr["target"] = target
        df_thr["roc_auc"] = roc
        df_thr["pr_auc"] = pr
        results_all.append(df_thr)

    out_df = pd.concat(results_all, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nAlle Threshold-Ergebnisse gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()