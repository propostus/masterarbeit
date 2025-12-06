# scripts/train_cv23_binary_threshold_sweep_mlp_tuned.py
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm


# -------------------------------------------------------
# MLP-Definition
# -------------------------------------------------------
class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # Binary-Logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,)


# -------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------
def make_loaders(X, y, groups, batch_size=512, random_state=42):
    """
    GroupShuffleSplit nach client_id: Train, Val, Test.
    """
    gss_outer = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=random_state)
    train_val_idx, test_idx = next(gss_outer.split(X, y, groups))

    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]
    groups_train_val = groups[train_val_idx]

    gss_inner = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=random_state)
    train_idx, val_idx = next(gss_inner.split(X_train_val, y_train_val, groups_train_val))

    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    # Skalierung nur auf Train fitten
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_scaled).float(),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_scaled).float(),
        torch.from_numpy(y_val.astype(np.float32)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test_scaled).float(),
        torch.from_numpy(y_test.astype(np.float32)),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    split_info = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "scaler": scaler,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    return train_loader, val_loader, test_loader, split_info


def collect_probs(model, loader, device):
    model.eval()
    probs_list = []
    ys_list = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            ys_list.append(yb.numpy())
    if not probs_list:
        return None, None
    probs = np.concatenate(probs_list)
    ys = np.concatenate(ys_list)
    return probs, ys


def tune_threshold_for_f1(y_true, probs, min_precision=0.5, step=0.05):
    """
    Sweep über feste Thresholds (0.05, 0.10, ..., 0.95)
    und wähle jene mit maximalem F1 bei precision >= min_precision.
    """
    best = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "acc": 0.0,
    }

    thresholds = np.arange(step, 1.0, step)
    for thr in thresholds:
        y_pred = (probs >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        if prec < min_precision:
            continue

        if f1 > best["f1"]:
            best.update(
                {
                    "threshold": float(thr),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "acc": float(acc),
                }
            )

    return best


def train_one_config(
    X,
    y,
    groups,
    input_dim,
    hidden_sizes,
    dropout,
    lr,
    batch_size,
    weight_decay,
    max_epochs,
    patience,
    device,
    random_state,
):
    """
    Trainiert eine MLP-Konfiguration, wählt bestes Epoch basierend auf Val-PR-AUC,
    und tuned anschließend den Threshold (F1 bei min_precision).
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    train_loader, val_loader, test_loader, split_info = make_loaders(
        X, y, groups, batch_size=batch_size, random_state=random_state
    )

    y_train = split_info["y_train"]
    y_val = split_info["y_val"]
    y_test = split_info["y_test"]

    # Klassenverteilung für pos_weight
    cnt = Counter(y_train)
    neg = cnt.get(0, 1)
    pos = cnt.get(1, 1)
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)

    model = SmallMLP(input_dim, hidden_sizes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val_pr_auc = -np.inf
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation: PR-AUC
        val_probs, y_val_probs = collect_probs(model, val_loader, device)
        if val_probs is None:
            break
        pr_auc = average_precision_score(y_val_probs, val_probs)
        roc_auc = roc_auc_score(y_val_probs, val_probs)

        print(
            f"    Epoch {epoch:02d}: train_loss={np.mean(epoch_losses):.4f}, "
            f"val_pr_auc={pr_auc:.4f}, val_roc_auc={roc_auc:.4f}"
        )

        if pr_auc > best_val_pr_auc + 1e-4:
            best_val_pr_auc = pr_auc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("    Early stopping.")
                break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    # Threshold-Tuning auf Validation
    val_probs, y_val_probs = collect_probs(model, val_loader, device)
    best_thr_info = tune_threshold_for_f1(
        y_val_probs, val_probs, min_precision=0.5, step=0.05
    )

    # Fallback, falls nichts die min_precision erfüllt hat
    if best_thr_info["f1"] == 0.0:
        # nehme 0.5 als Default
        thr = 0.5
        y_pred_val = (val_probs >= thr).astype(int)
        best_thr_info.update(
            {
                "threshold": thr,
                "precision": precision_score(y_val_probs, y_pred_val, zero_division=0),
                "recall": recall_score(y_val_probs, y_pred_val, zero_division=0),
                "f1": f1_score(y_val_probs, y_pred_val, zero_division=0),
                "acc": accuracy_score(y_val_probs, y_pred_val),
            }
        )

    # Val-Gesamtmetriken
    val_roc_auc = roc_auc_score(y_val_probs, val_probs)
    val_pr_auc = average_precision_score(y_val_probs, val_probs)

    # Test-Metriken mit diesem Threshold
    test_probs, y_test_probs = collect_probs(model, test_loader, device)
    thr = best_thr_info["threshold"]
    y_pred_test = (test_probs >= thr).astype(int)

    test_acc = accuracy_score(y_test_probs, y_pred_test)
    test_prec = precision_score(y_test_probs, y_pred_test, zero_division=0)
    test_rec = recall_score(y_test_probs, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test_probs, y_pred_test, zero_division=0)
    test_roc_auc = roc_auc_score(y_test_probs, test_probs)
    test_pr_auc = average_precision_score(y_test_probs, test_probs)

    result = {
        # Validation-Summary
        "val_best_pr_auc": float(best_val_pr_auc),
        "val_roc_auc": float(val_roc_auc),
        "val_pr_auc": float(val_pr_auc),
        "val_threshold": best_thr_info["threshold"],
        "val_precision": best_thr_info["precision"],
        "val_recall": best_thr_info["recall"],
        "val_f1": best_thr_info["f1"],
        "val_acc": best_thr_info["acc"],
        # Test
        "test_threshold": thr,
        "test_acc": float(test_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
    }

    return result


# -------------------------------------------------------
# Hauptfunktion
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MLP-Binary-Klassifikation für verschiedene WER-Schwellen (mit Hyperparameter-Tuning)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Pfad zu merged_sigmos_wavlm_cv23_balanced_multiwer.csv",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Pfad zur Ergebnis-CSV",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu|cuda|mps|auto",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=40,
        help="Maximale Epochen pro Konfiguration",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early-Stopping-Patience",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batchgröße (wird durch Grid ggf. überschrieben)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20],
        help="WER-Schwellen für die Definition der Positivklasse",
    )
    args = parser.parse_args()

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
    print(f"Lade Datensatz von: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"Shape: {df.shape}")

    # Feature-Spalten: alles numerisch außer filename/client_id/age/gender/sentence
    non_feature_cols = {
        "filename",
        "client_id",
        "age",
        "gender",
        "sentence",
        "wer_tiny",
        "wer_base",
        "wer_small",
    }
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols].values.astype(np.float32)
    groups = df["client_id"].astype(str).values

    targets = ["wer_tiny", "wer_base", "wer_small"]

    # Hyperparameter-Grid
    grid = []
    hidden_options = [[512, 256], [512, 256, 128]]
    dropout_options = [0.2, 0.3]
    lr_options = [1e-3, 5e-4]
    batch_options = [256, 512]
    weight_decay_options = [0.0, 1e-4]

    for hs in hidden_options:
        for dr in dropout_options:
            for lr in lr_options:
                for bs in batch_options:
                    for wd in weight_decay_options:
                        grid.append(
                            {
                                "hidden_sizes": hs,
                                "dropout": dr,
                                "lr": lr,
                                "batch_size": bs,
                                "weight_decay": wd,
                            }
                        )

    print(f"Hyperparameter-Kombinationen: {len(grid)}")

    results = []

    for target in targets:
        wer = df[target].values.astype(np.float32)
        mean_wer = wer.mean()
        median_wer = float(np.median(wer))
        print("\n" + "=" * 70)
        print(f"Zielvariable: {target}")
        print(f"WER-Statistik: mean={mean_wer:.4f}, median={median_wer:.4f}")
        print("=" * 70)

        for thr in args.thresholds:
            print("\n" + "-" * 70)
            print(f"Schwelle: {target} <= {thr:.2f} als Klasse 1")
            print("-" * 70)

            y = (wer <= thr).astype(int)
            cnt = Counter(y)
            print(f"Label-Verteilung (gesamt) für {target}, thr={thr}: {cnt}")

            best_cfg = None
            best_val_f1 = -np.inf
            best_result = None

            for cfg_id, cfg in enumerate(grid, start=1):
                print(
                    f"\nConfig {cfg_id}/{len(grid)}: "
                    f"hidden={cfg['hidden_sizes']}, "
                    f"dropout={cfg['dropout']}, lr={cfg['lr']}, "
                    f"batch={cfg['batch_size']}, wd={cfg['weight_decay']}"
                )

                res = train_one_config(
                    X=X,
                    y=y,
                    groups=groups,
                    input_dim=X.shape[1],
                    hidden_sizes=cfg["hidden_sizes"],
                    dropout=cfg["dropout"],
                    lr=cfg["lr"],
                    batch_size=cfg["batch_size"],
                    weight_decay=cfg["weight_decay"],
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    device=device,
                    random_state=args.random_state,
                )

                print(
                    f"  -> val_f1={res['val_f1']:.3f}, "
                    f"val_prec={res['val_precision']:.3f}, "
                    f"val_rec={res['val_recall']:.3f}, "
                    f"val_pr_auc={res['val_pr_auc']:.3f}"
                )

                if res["val_f1"] > best_val_f1:
                    best_val_f1 = res["val_f1"]
                    best_cfg = cfg
                    best_result = res

            print("\nBeste Konfiguration für "
                  f"{target}, thr={thr}: val_f1={best_val_f1:.3f}")
            print(best_cfg)

            row = {
                "model": "MLP_small_binary_tuned",
                "target": target,
                "wer_threshold": thr,
            }
            row.update(best_cfg)
            row.update(best_result)
            results.append(row)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    print(f"\nAlle Ergebnisse gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()