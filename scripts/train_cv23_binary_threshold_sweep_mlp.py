# scripts/train_cv23_binary_threshold_sweep_mlp.py
"""
Trainiert kleine MLP-Binärklassifikationsmodelle für verschiedene WER-Schwellen
(z.B. 0.05, 0.10, 0.20) auf dem CV23-balanced-Datensatz.

- Ziele: wer_tiny, wer_base, wer_small
- Labels: y = 1, wenn wer <= threshold, sonst 0
- Features: SigMOS + WavLM (alle numerischen Spalten, WERs & Meta werden gedroppt)
- Split: gruppenbasiert nach client_id in Train / Val / Test
- Modell: kleines MLP mit 2 Hidden-Layern
- Optimierung: BCEWithLogitsLoss mit pos_weight, Early Stopping auf Val-ROC-AUC
- Pro Threshold: beste Prob.-Schwelle wird auf Val gesucht
"""

import argparse
import os
from collections import Counter
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 512, hidden2: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),  # Binary Logit
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_group_splits(
    n_samples: int,
    groups: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gibt drei Indize-Arrays für Train / Val / Test zurück,
    gruppenbasiert nach 'groups' (z.B. client_id).
    """
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss_outer.split(np.arange(n_samples), groups=groups))

    # Relativer Val-Anteil im Train+Val-Block
    rel_val_size = val_size / (1.0 - test_size)
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=random_state + 1)
    train_idx, val_idx = next(gss_inner.split(trainval_idx, groups=groups[trainval_idx]))

    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    return train_idx, val_idx, test_idx


def train_mlp_one_task(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    max_epochs: int = 40,
    patience: int = 6,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Trainiert ein MLP für einen Binärtask und gibt Test- und Val-Metriken zurück.
    """

    # Split
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Skaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Tensors / Loader
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float()

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    # Modell
    input_dim = X_train.shape[1]
    model = SmallMLP(input_dim=input_dim).to(device)

    # Klassenschiefe für pos_weight
    pos_count = float((y_train == 1).sum())
    neg_count = float((y_train == 0).sum())
    pos_weight = torch.tensor(neg_count / max(pos_count, 1.0), dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = -np.inf
    best_state = None
    epochs_no_improve = 0

    # Training
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_logits_list = []
        val_y_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_logits_list.append(logits.cpu().numpy())
                val_y_list.append(yb.cpu().numpy())
        val_logits = np.concatenate(val_logits_list)
        val_y = np.concatenate(val_y_list)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        # ROC-AUC für Early Stopping
        try:
            val_roc_auc = roc_auc_score(val_y, val_probs)
        except ValueError:
            val_roc_auc = np.nan

        print(f"  Epoch {epoch:02d}: train_loss={np.mean(train_losses):.4f}, val_roc_auc={val_roc_auc:.4f}")

        if val_roc_auc > best_val_auc:
            best_val_auc = val_roc_auc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("  Early stopping.")
                break

    # Bestes Modell laden
    if best_state is not None:
        model.load_state_dict(best_state)

    # --------------------------------------------------
    # Threshold-Suche auf Val (Präzision hoch, Recall>=0.1)
    # --------------------------------------------------
    model.eval()

    def collect_probs(loader):
        probs_list = []
        ys_list = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb).cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))
                probs_list.append(probs)
                ys_list.append(yb.numpy())
        return np.concatenate(probs_list), np.concatenate(ys_list)

    val_probs, val_y = collect_probs(val_loader)
    test_probs, test_y = collect_probs(test_loader)

    thresholds = np.linspace(0.1, 0.9, 17)
    best_row = None
    best_key = -np.inf

    for thr in thresholds:
        val_pred = (val_probs >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            val_y, val_pred, average="binary", zero_division=0
        )
        acc = accuracy_score(val_y, val_pred)

        # wir wollen hohe Präzision, aber Recall nicht komplett 0
        if rec < 0.10:
            continue

        key = prec  # primär nach Präzision sortieren
        if key > best_key:
            best_key = key
            best_row = {
                "val_threshold": float(thr),
                "val_precision": float(prec),
                "val_recall": float(rec),
                "val_f1": float(f1),
                "val_acc": float(acc),
            }

    if best_row is None:
        # Fallback: Schwelle 0.5
        thr = 0.5
        val_pred = (val_probs >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            val_y, val_pred, average="binary", zero_division=0
        )
        acc = accuracy_score(val_y, val_pred)
        best_row = {
            "val_threshold": 0.5,
            "val_precision": float(prec),
            "val_recall": float(rec),
            "val_f1": float(f1),
            "val_acc": float(acc),
        }

    best_thr = best_row["val_threshold"]

    # --------------------------------------------------
    # Test-Metriken mit dieser Schwelle
    # --------------------------------------------------
    test_pred = (test_probs >= best_thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_y, test_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(test_y, test_pred)
    try:
        roc_auc = roc_auc_score(test_y, test_probs)
    except ValueError:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(test_y, test_probs)
    except ValueError:
        pr_auc = np.nan

    test_metrics = {
        "test_threshold": float(best_thr),
        "test_acc": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
    }

    val_metrics = best_row
    return val_metrics, test_metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Trainiert kleine MLP-Binärmodelle für verschiedene WER-Schwellen (5/10/20%)."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="CSV mit SigMOS+WavLM+WER (z.B. merged_sigmos_wavlm_cv23_balanced_multiwer.csv)")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Pfad zur Ergebnis-CSV")
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu | cuda | mps | auto")
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20],
                        help="WER-Schwellen, z.B. 0.05 0.10 0.20")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Verwende Gerät: {device}")

    print(f"Lade Datensatz von: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"Shape: {df.shape}")

    target_cols = ["wer_tiny", "wer_base", "wer_small"]
    meta_cols = ["filename", "client_id", "age", "gender", "sentence"]

    # Features: alles numerische außer Targets und Meta
    feature_df = df.drop(columns=target_cols + meta_cols, errors="ignore")
    feature_df = feature_df.select_dtypes(include=[np.number])
    X_all = feature_df.values.astype(np.float32)
    print(f"Anzahl Feature-Spalten: {X_all.shape[1]}")

    # Gruppen für Splits
    if "client_id" in df.columns:
        groups = df["client_id"].astype(str).values
    else:
        groups = np.arange(len(df))

    n_samples = len(df)
    train_idx, val_idx, test_idx = make_group_splits(
        n_samples=n_samples,
        groups=groups,
        test_size=0.2,
        val_size=0.2,
        random_state=args.random_state,
    )

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    all_results = []

    for target in target_cols:
        if target not in df.columns:
            print(f"Warnung: Spalte {target} nicht im Datensatz, überspringe.")
            continue

        y_reg = df[target].values.astype(np.float32)
        print("\n" + "=" * 70)
        print(f"Zielvariable: {target}")
        print("=" * 70)
        print(f"WER-Statistik: mean={y_reg.mean():.4f}, median={np.median(y_reg):.4f}")

        for thr in args.thresholds:
            print("\n" + "-" * 70)
            print(f"Schwelle: WER <= {thr:.2f} als Klasse 1")
            print("-" * 70)

            y_bin = (y_reg <= thr).astype(int)

            # Label-Verteilung
            train_labels = y_bin[train_idx]
            cnt = Counter(train_labels.tolist())
            print(f"Label-Verteilung (Train) für {target}, thr={thr}: {cnt}")

            val_metrics, test_metrics = train_mlp_one_task(
                X_all,
                y_bin,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                device=device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
            )

            result_row = {
                "model": "MLP_small_binary",
                "target": target,
                "wer_threshold": float(thr),
                **val_metrics,
                **test_metrics,
            }
            all_results.append(result_row)

    results_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)
    print(f"\nGesamtergebnisse gespeichert unter: {args.out_csv}")
    print(results_df)


if __name__ == "__main__":
    main()