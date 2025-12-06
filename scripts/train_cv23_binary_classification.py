# scripts/train_cv23_binary_classification.py
# -------------------------------------------
# Binary-Klassifikation: "korrekt" (WER == 0) vs. "fehlerhaft" (WER > 0)
# für wer_tiny, wer_base, wer_small.
# Modelle:
#   - LightGBM
#   - CatBoost
#   - (optionales) MLP mit Mini-Batches + Early Stopping
# -------------------------------------------

import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------
# Hilfsfunktionen für MLP
# ---------------------------------------------------------------------


class MLPBinary(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))  # Logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_binary(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    max_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    hidden_sizes=(256, 128),
    dropout: float = 0.2,
    patience: int = 3,
):
    """Trainiert ein kleines MLP mit Early Stopping und gibt bestes Modell + Scaler zurück."""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_t = torch.from_numpy(X_train_scaled.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    X_val_t = torch.from_numpy(X_val_scaled.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.float32))

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLPBinary(X_train.shape[1], hidden_sizes=hidden_sizes, dropout=dropout).to(
        device
    )

    # class imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight_value = float(neg_count / max(pos_count, 1))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -np.inf
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

        # Validation
        model.eval()
        val_losses = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(yb.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        preds_bin = (all_probs >= 0.5).astype(int)

        val_f1 = f1_score(all_labels, preds_bin)
        avg_train_loss = float(np.mean(train_losses))
        avg_val_loss = float(np.mean(val_losses))

        print(
            f"    Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} "
            f"| val_loss={avg_val_loss:.4f} | val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early Stopping nach {epoch} Epochen (best val_f1={best_val_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler


def eval_binary_probs(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = np.nan
    try:
        pr = average_precision_score(y_true, y_prob)
    except ValueError:
        pr = np.nan
    return acc, f1, roc, pr


# ---------------------------------------------------------------------
# Haupt-Training
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Binary Klassifikation: korrekte Transkription (WER == 0) vs. fehlerhaft (WER > 0)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="results/datasets/merged_sigmos_wavlm_cv23_balanced_multiwer.csv",
        help="Pfad zur Trainings-CSV (SigMOS + WavLM + WER)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/model_comparisons/binary_cv23_baseline.csv",
        help="Pfad zur Ergebnis-CSV",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Grenze für 'korrekt' (z.B. 0.0, 0.05)",
    )
    parser.add_argument(
        "--skip_mlp",
        action="store_true",
        help="MLP-Training überspringen (nur LightGBM & CatBoost)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu | cuda | mps | auto",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Gerät
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
    df = pd.read_csv(args.dataset)
    print("Shape Datensatz:", df.shape)

    # Feature-Spalten (numerische Spalten außer Targets & Metadaten)
    drop_cols = {
        "wer_tiny",
        "wer_base",
        "wer_small",
        "filename",
        "client_id",
        "age",
        "gender",
        "sentence",
        "reference",
        "hypothesis",
    }
    feature_cols = [
        c
        for c in df.columns
        if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    print(f"Anzahl Feature-Spalten: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    groups = df["client_id"].astype(str).values

    # Targets
    wer_targets = ["wer_tiny", "wer_base", "wer_small"]

    results_rows = []

    # Gruppierter Split (Speaker-Level)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, groups=groups))
    X_train_full, X_test_full = X[train_idx], X[test_idx]
    groups_train = groups[train_idx]

    for target in wer_targets:
        print("\n" + "=" * 70)
        print(f"Zielvariable: {target} (WER <= {args.threshold} als Klasse 1)")
        print("=" * 70)

        y = df[target].values
        y_bin = (y <= args.threshold).astype(int)

        y_train_full = y_bin[train_idx]
        y_test = y_bin[test_idx]

        # innerer Split: Train / Val für MLP und evtl. für LGBM
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, val_idx = next(gss_inner.split(X_train_full, groups=groups_train))
        X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
        y_tr, y_val = y_train_full[tr_idx], y_train_full[val_idx]

        print(
            "Label-Verteilung (Train) für",
            target,
            ":",
            Counter(y_tr),
        )

        # ---------------- LightGBM ----------------
        print(f"\nTrainiere LightGBM für {target} ...")
        lgbm = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        lgbm.fit(X_tr, y_tr)
        prob_test = lgbm.predict_proba(X_test_full)[:, 1]
        acc, f1, roc, pr = eval_binary_probs(y_test, prob_test)
        print(
            f"  LightGBM Test-Metriken für {target}: "
            f"acc={acc:.3f}, f1={f1:.3f}, roc_auc={roc:.3f}, pr_auc={pr:.3f}"
        )
        results_rows.append(
            {
                "model": "LightGBM_binary",
                "target": target,
                "threshold": args.threshold,
                "acc": acc,
                "f1": f1,
                "roc_auc": roc,
                "pr_auc": pr,
            }
        )

        # ---------------- CatBoost ----------------
        print(f"\nTrainiere CatBoost für {target} ...")
        cat = CatBoostClassifier(
            iterations=400,
            learning_rate=0.03,
            depth=8,
            loss_function="Logloss",
            verbose=False,
            random_state=42,
        )
        cat.fit(X_tr, y_tr)
        prob_test = cat.predict_proba(X_test_full)[:, 1]
        acc, f1, roc, pr = eval_binary_probs(y_test, prob_test)
        print(
            f"  CatBoost Test-Metriken für {target}: "
            f"acc={acc:.3f}, f1={f1:.3f}, roc_auc={roc:.3f}, pr_auc={pr:.3f}"
        )
        results_rows.append(
            {
                "model": "CatBoost_binary",
                "target": target,
                "threshold": args.threshold,
                "acc": acc,
                "f1": f1,
                "roc_auc": roc,
                "pr_auc": pr,
            }
        )

        # ---------------- MLP ----------------
        if not args.skip_mlp:
            print(f"\nTrainiere MLP für {target} ...")
            model_mlp, scaler = train_mlp_binary(
                X_tr,
                y_tr,
                X_val,
                y_val,
                device=device,
                max_epochs=20,
                batch_size=512,
                lr=1e-3,
                hidden_sizes=(256, 128),
                dropout=0.2,
                patience=3,
            )

            # Test-Eval
            X_test_scaled = scaler.transform(X_test_full)
            X_test_t = torch.from_numpy(X_test_scaled.astype(np.float32)).to(device)
            model_mlp.eval()
            with torch.no_grad():
                logits = model_mlp(X_test_t)
                probs = torch.sigmoid(logits).cpu().numpy()
            acc, f1, roc, pr = eval_binary_probs(y_test, probs)
            print(
                f"  MLP Test-Metriken für {target}: "
                f"acc={acc:.3f}, f1={f1:.3f}, roc_auc={roc:.3f}, pr_auc={pr:.3f}"
            )
            results_rows.append(
                {
                    "model": "MLP_binary",
                    "target": target,
                    "threshold": args.threshold,
                    "acc": acc,
                    "f1": f1,
                    "roc_auc": roc,
                    "pr_auc": pr,
                }
            )

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(args.out_csv, index=False)
    print("\nFertig. Ergebnisse gespeichert unter:", args.out_csv)
    print(results_df)


if __name__ == "__main__":
    main()