# scripts/evaluate_cv23_binary_mlp_on_unseen.py
# ------------------------------------------------------
# Trainiert pro Zielvariable ein kleines MLP für die
# Klassifikation "perfekt transkribiert" (WER == 0.0)
# auf dem CV23-balanced Datensatz und evaluiert das
# Modell anschließend auf dem Unseen-Datensatz.
#
# Ausgabe: CSV mit Metriken auf dem Unseen-Set.
# ------------------------------------------------------

import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Wählt Feature-Spalten (nur numerische, ohne Targets/Metadaten)."""
    exclude = {
        "filename",
        "wer_tiny",
        "wer_base",
        "wer_small",
        "client_id",
        "age",
        "gender",
        "sentence",
        "reference",
        "hypothesis",
        "source",
        "group_id",
    }
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and np.issubdtype(df[c].dtype, np.number)
    ]
    return feature_cols


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_group_splits(df: pd.DataFrame, random_state: int = 42):
    """Erstellt (train, val, test)-Indices mit Speaker-Gruppierung."""
    groups = df["client_id"].astype(str)

    gss = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=random_state
    )
    train_val_idx, test_idx = next(gss.split(df, groups=groups))

    df_train_val = df.iloc[train_val_idx]
    groups_tv = df_train_val["client_id"].astype(str)

    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=random_state + 1
    )
    train_idx_rel, val_idx_rel = next(gss2.split(df_train_val, groups=groups_tv))

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    return train_idx, val_idx, test_idx


def train_mlp_binary(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    max_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 5,
):
    """Trainiert ein kleines MLP für binäre Klassifikation, gibt bestes Modell und Val-Probs zurück."""

    input_dim = X_train.shape[1]
    model = SmallMLP(input_dim=input_dim).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.float32))
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # class imbalance handling
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = neg_count / max(pos_count, 1.0)
    pos_weight_tensor = torch.tensor(
        [pos_weight], dtype=torch.float32, device=device
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = np.inf
    best_state = None
    epochs_no_improve = 0

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

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(
            f"  Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

        if val_loss + 1e-4 < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("  Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Val-Probs für Threshold-Tuning
    model.eval()
    val_probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            val_probs.append(probs.cpu().numpy())
    val_probs = np.concatenate(val_probs, axis=0)

    return model, val_probs


def tune_threshold(val_probs, y_val, min_recall: float = 0.1):
    """Grid-Search über Thresholds, Ziel: maximale Precision bei recall >= min_recall."""
    thresholds = np.linspace(0.1, 0.9, 17)
    best = None

    for thr in thresholds:
        y_pred = (val_probs >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary", zero_division=0
        )
        acc = accuracy_score(y_val, y_pred)

        if rec < min_recall:
            continue

        cand = {
            "threshold": thr,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "acc": acc,
        }

        if best is None or cand["precision"] > best["precision"]:
            best = cand

    if best is None:
        # Fallback: beste F1 ohne Recall-Bedingung
        for thr in thresholds:
            y_pred = (val_probs >= thr).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average="binary", zero_division=0
            )
            acc = accuracy_score(y_val, y_pred)
            cand = {
                "threshold": thr,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "acc": acc,
            }
            if best is None or cand["f1"] > best["f1"]:
                best = cand

    return best


def eval_on_split(probs, y_true, threshold: float):
    """Berechnet Metriken (acc, f1, precision, recall, roc_auc, pr_auc) für gegebenen Threshold."""
    y_pred = (probs >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan
    pr = average_precision_score(y_true, probs)
    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr,
    }


# ------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Trainiere kleines MLP für WER==0-Klassifikation (CV23) und evaluiere auf Unseen."
    )
    parser.add_argument(
        "--train_csv",
        required=True,
        help="CV23 balanced multiwer CSV (mit wer_tiny/wer_base/wer_small).",
    )
    parser.add_argument(
        "--unseen_csv",
        required=True,
        help="Unseen CSV (delta 22) mit gleichen Spalten.",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Pfad für Ergebnis-CSV (Unseen-Metriken).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Rechengerät.",
    )
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # Gerät wählen
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Verwende Gerät: {device}")

    # Train-Daten laden
    print(f"Lade Train-Daten von: {args.train_csv}")
    df_train_full = pd.read_csv(args.train_csv)
    print(f"Shape Train: {df_train_full.shape}")

    # Client-ID sicherstellen
    if "client_id" not in df_train_full.columns:
        raise RuntimeError("Spalte 'client_id' wird für Group-Split benötigt.")

    # Features wählen
    feature_cols = select_feature_columns(df_train_full)
    print(f"Anzahl Feature-Spalten: {len(feature_cols)}")

    # Splits
    train_idx, val_idx, test_idx = make_group_splits(
        df_train_full, random_state=args.random_state
    )

    X = df_train_full[feature_cols].values.astype(np.float32)

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    # StandardScaler nur auf Train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Unseen laden
    print(f"Lade Unseen-Daten von: {args.unseen_csv}")
    df_unseen = pd.read_csv(args.unseen_csv)
    print(f"Shape Unseen: {df_unseen.shape}")

    # sicherstellen, dass alle Feature-Spalten existieren
    missing_feats = [c for c in feature_cols if c not in df_unseen.columns]
    if missing_feats:
        raise RuntimeError(
            f"Folgende Feature-Spalten fehlen in Unseen-CSV: {missing_feats}"
        )

    X_unseen = df_unseen[feature_cols].values.astype(np.float32)
    X_unseen = scaler.transform(X_unseen)

    targets = ["wer_tiny", "wer_base", "wer_small"]
    results = []

    for target in targets:
        print("=" * 70)
        print(f"Zielvariable: {target} (WER == 0.0 als Klasse 1)")
        print("=" * 70)

        y_all = (df_train_full[target].values == 0.0).astype(int)

        y_train = y_all[train_idx]
        y_val = y_all[val_idx]
        y_test = y_all[test_idx]

        print(
            f"Label-Verteilung (Train) für {target}: {Counter(y_train)}"
        )

        # MLP trainieren
        model, val_probs = train_mlp_binary(
            X_train,
            y_train,
            X_val,
            y_val,
            device=device,
            max_epochs=args.max_epochs,
            batch_size=512,
            lr=1e-3,
            patience=args.patience,
        )

        # Threshold-Tuning auf Validation
        best_thr = tune_threshold(val_probs, y_val, min_recall=0.1)
        print("\nBeste Schwelle (Validation):")
        for k, v in best_thr.items():
            print(f"  {k} = {v:.4f}")
        thr = float(best_thr["threshold"])

        # Test-Set-Probs
        model.eval()
        with torch.no_grad():
            test_logits = model(torch.from_numpy(X_test).to(device))
            test_probs = torch.sigmoid(test_logits).cpu().numpy()

        test_metrics = eval_on_split(test_probs, y_test, thr)
        print(
            f"\nTest-Metriken für {target}: "
            f"acc={test_metrics['acc']:.3f}, "
            f"f1={test_metrics['f1']:.3f}, "
            f"roc_auc={test_metrics['roc_auc']:.3f}, "
            f"pr_auc={test_metrics['pr_auc']:.3f}"
        )

        # Unseen-Probs
        with torch.no_grad():
            unseen_logits = model(torch.from_numpy(X_unseen).to(device))
            unseen_probs = torch.sigmoid(unseen_logits).cpu().numpy()

        y_unseen = (df_unseen[target].values == 0.0).astype(int)
        unseen_metrics = eval_on_split(unseen_probs, y_unseen, thr)
        print(
            f"\nUNSEEN-Metriken für {target}: "
            f"acc={unseen_metrics['acc']:.3f}, "
            f"f1={unseen_metrics['f1']:.3f}, "
            f"roc_auc={unseen_metrics['roc_auc']:.3f}, "
            f"pr_auc={unseen_metrics['pr_auc']:.3f}"
        )

        row = {
            "model": "MLP_small_binary",
            "target": target,
            "threshold": thr,
        }
        for prefix, m in [("val_best_", best_thr), ("test_", test_metrics), ("unseen_", unseen_metrics)]:
            for k, v in m.items():
                row[f"{prefix}{k}"] = float(v)

        results.append(row)

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)
    print("\n=== Fertig. Ergebnisse gespeichert unter:", args.out_csv)
    print(results_df)


if __name__ == "__main__":
    main()