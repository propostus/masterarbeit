# scripts/eval_binary_mlp_thresholds_on_unseen_fixed.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------
# Einfaches MLP
# --------------------------------------------------------
class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout=0.3):
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


# --------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------
def get_device(name: str):
    name = name.lower()
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Nur numerische Features, ohne WER-Spalten."""
    drop_targets = ["wer_tiny", "wer_base", "wer_small"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in drop_targets]
    return feature_cols


def make_group_splits(df: pd.DataFrame, group_col="client_id",
                      test_size=0.2, val_size=0.2, random_state=42):
    """Group-basiert: erst Train+Val vs Test, dann Train vs Val."""
    groups = df[group_col].values

    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    trainval_idx, test_idx = next(gss.split(df, groups=groups))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    groups_trainval = df_trainval[group_col].values
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state + 1
    )
    train_idx, val_idx = next(
        gss2.split(df_trainval, groups=groups_trainval)
    )

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val, df_test


def make_loaders(X_tr, y_tr, X_val, y_val, X_te, y_te,
                 batch_size: int):
    def make_tensor_dataset(X, y):
        if X is None or y is None:
            return None
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))
        ds = TensorDataset(X_t, y_t)
        return ds

    ds_tr = make_tensor_dataset(X_tr, y_tr)
    ds_val = make_tensor_dataset(X_val, y_val)
    ds_te = make_tensor_dataset(X_te, y_te)

    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True) if ds_tr else None
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False) if ds_val else None
    loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False) if ds_te else None
    return loader_tr, loader_val, loader_te


def collect_probs(loader, model, device):
    model.eval()
    all_probs = []
    all_y = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_y.append(yb.cpu().numpy())
    if not all_probs:
        return np.array([]), np.array([])
    return np.concatenate(all_probs), np.concatenate(all_y)


def compute_metrics(y_true, probs, thr_prob: float):
    """Berechnet Acc/Prec/Rec/F1/ROC/PR für einen festen Wahrscheinlichkeits-Threshold."""
    if probs.size == 0:
        return {
            "acc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
        }

    preds = (probs >= thr_prob).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, preds)

    try:
        roc_auc = roc_auc_score(y_true, probs)
    except ValueError:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_true, probs)
    except ValueError:
        pr_auc = np.nan

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# --------------------------------------------------------
# Training & Evaluation für eine Konfiguration
# --------------------------------------------------------
def train_and_eval_one(
    df_all: pd.DataFrame,
    df_unseen: pd.DataFrame,
    target: str,
    wer_threshold: float,
    hidden_sizes,
    dropout: float,
    lr: float,
    batch_size: int,
    weight_decay: float,
    prob_threshold: float,
    device,
    max_epochs: int,
    patience: int,
    random_state: int,
):
    # 1) Labels bauen: WER <= wer_threshold → Klasse 1
    if target not in df_all.columns:
        raise ValueError(f"Target {target} nicht im Trainings-DataFrame.")

    y_all = (df_all[target].values <= wer_threshold).astype(int)

    # Group-Split
    df_train, df_val, df_test = make_group_splits(
        df_all, group_col="client_id",
        test_size=0.2, val_size=0.25,  # 0.8 -> 0.6 train, 0.2 val, 0.2 test
        random_state=random_state,
    )

    # Labels für Splits
    y_tr = (df_train[target].values <= wer_threshold).astype(int)
    y_val = (df_val[target].values <= wer_threshold).astype(int)
    y_te = (df_test[target].values <= wer_threshold).astype(int)

    # Features
    feature_cols = get_feature_cols(df_all)
    X_tr = df_train[feature_cols].values.astype(np.float32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    X_te = df_test[feature_cols].values.astype(np.float32)

    # Unseen-Labels und -Features
    if target not in df_unseen.columns:
        raise ValueError(f"Target {target} nicht im Unseen-DataFrame.")
    y_unseen = (df_unseen[target].values <= wer_threshold).astype(int)
    X_unseen = df_unseen[feature_cols].values.astype(np.float32)

    # Skalierung
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)
    X_unseen = scaler.transform(X_unseen)

    # Loader
    train_loader, val_loader, test_loader = make_loaders(
        X_tr, y_tr, X_val, y_val, X_te, y_te, batch_size=batch_size
    )
    unseen_ds = TensorDataset(
        torch.from_numpy(X_unseen.astype(np.float32)),
        torch.from_numpy(y_unseen.astype(np.float32)),
    )
    unseen_loader = DataLoader(unseen_ds, batch_size=batch_size, shuffle=False)

    # Modell
    torch.manual_seed(random_state)
    model = SmallMLP(
        input_dim=X_tr.shape[1],
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_pr_auc = -np.inf
    best_state_dict = None
    epochs_no_improve = 0

    # 2) Training
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

        # Validierung
        val_probs, val_y_true = collect_probs(val_loader, model, device)
        val_metrics = compute_metrics(val_y_true, val_probs, prob_threshold)
        val_pr_auc = val_metrics["pr_auc"]

        print(
            f"  Epoch {epoch:02d}: train_loss={np.mean(train_losses):.4f}, "
            f"val_pr_auc={val_pr_auc:.4f}, val_roc_auc={val_metrics['roc_auc']:.4f}"
        )

        # Early Stopping auf Basis PR-AUC
        if val_pr_auc > best_val_pr_auc + 1e-4:
            best_val_pr_auc = val_pr_auc
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("  Early stopping.")
                break

    # Bestes Modell laden
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 3) Final Test- und Unseen-Metriken
    test_probs, test_y_true = collect_probs(test_loader, model, device)
    unseen_probs, unseen_y_true = collect_probs(unseen_loader, model, device)

    test_metrics = compute_metrics(test_y_true, test_probs, prob_threshold)
    unseen_metrics = compute_metrics(unseen_y_true, unseen_probs, prob_threshold)

    result = {
        "target": target,
        "wer_threshold": wer_threshold,
        "prob_threshold": prob_threshold,
        "hidden_sizes": str(hidden_sizes),
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "val_best_pr_auc": best_val_pr_auc,
        "test_acc": test_metrics["acc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "unseen_acc": unseen_metrics["acc"],
        "unseen_precision": unseen_metrics["precision"],
        "unseen_recall": unseen_metrics["recall"],
        "unseen_f1": unseen_metrics["f1"],
        "unseen_roc_auc": unseen_metrics["roc_auc"],
        "unseen_pr_auc": unseen_metrics["pr_auc"],
    }
    return result


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Eval MLP-Binary-Configs (WER-Thresholds) auf Unseen (fixed metrics)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Train/Val/Test-Datensatz (CV23 balanced multiwer).",
    )
    parser.add_argument(
        "--unseen_csv",
        type=str,
        required=True,
        help="Unseen-Datensatz mit WER-Spalten.",
    )
    parser.add_argument(
        "--config_csv",
        type=str,
        required=True,
        help="CSV mit getunten Configs (binary_mlp_threshold_sweep_tuned_5_10_20.csv).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Pfad für Ergebnis-CSV.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu | cuda | mps | auto",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=40,
        help="Maximale Epochen pro Config.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early-Stopping-Patience.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random-Seed für Splits/Initialisierung.",
    )

    args = parser.parse_args()
    device = get_device(args.device)
    print(f"Verwende Gerät: {device}")

    df_all = pd.read_csv(args.dataset)
    df_unseen = pd.read_csv(args.unseen_csv)
    cfg = pd.read_csv(args.config_csv)

    results = []

    for _, row in tqdm(cfg.iterrows(), total=len(cfg), desc="Configs"):
        target = row["target"]
        wer_thr = float(row["wer_threshold"])
        hidden_sizes = eval(row["hidden_sizes"])  # z.B. "[512, 256]"
        dropout = float(row["dropout"])
        lr = float(row["lr"])
        batch_size = int(row["batch_size"])
        weight_decay = float(row["weight_decay"])
        prob_threshold = float(row["val_threshold"])  # denselben Threshold verwenden

        print(
            f"\n=== {target} @ WER <= {wer_thr:.2f} "
            f"(hidden={hidden_sizes}, drop={dropout}, lr={lr}, "
            f"bs={batch_size}, wd={weight_decay}, prob_thr={prob_threshold}) ==="
        )

        res = train_and_eval_one(
            df_all=df_all,
            df_unseen=df_unseen,
            target=target,
            wer_threshold=wer_thr,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            prob_threshold=prob_threshold,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            random_state=args.random_state,
        )
        results.append(res)

    out_df = pd.DataFrame(results)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nFertig. Ergebnisse gespeichert unter: {args.out_csv}")
    print(out_df)


if __name__ == "__main__":
    main()