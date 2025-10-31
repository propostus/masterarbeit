# scripts/evaluate_unseen_multioutput_pytorch.py
import os
import argparse
import math
import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


TARGETS = ["wer_tiny", "wer_base", "wer_small"]


def stem_from_filename(fn: str) -> str:
    """Gruppen-ID aus Dateiname: clean + _snrX zusammenhalten."""
    base = os.path.basename(str(fn)).lower()
    if "." in base:
        base = base.rsplit(".", 1)[0]
    # typische SNR-Kennungen entfernen
    for tag in ["_snr0", "_snr10", "_snr20"]:
        base = base.replace(tag, "")
    return base


def r2_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(((y_true - y_pred) ** 2).mean())
    mae = np.abs(y_true - y_pred).mean()
    return r2, rmse, mae


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 3, hidden: List[int] = [512, 256, 128], p_drop: float = 0.15):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p_drop)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def build_xy(df: pd.DataFrame, use_snr: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Nicht als Features verwenden:
    drop_cols = ["filename"] + TARGETS
    # snr optional als Feature
    if not use_snr and "snr" in df.columns:
        drop_cols.append("snr")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df[TARGETS].copy()

    # Nur numerische Spalten in X behalten
    X = X.select_dtypes(include=[np.number])

    return X, y


def split_grouped(df: pd.DataFrame, test_size: float, val_size: float, random_state: int):
    groups = df["group_id"].values

    # zuerst Train+Val vs Test
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_trainval, idx_test = next(gss_outer.split(df, groups=groups))

    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    # dann Train vs Val innerhalb Trainval
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=random_state)
    groups_tv = df_trainval["group_id"].values
    idx_train, idx_val = next(gss_inner.split(df_trainval, groups=groups_tv))

    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val = df_trainval.iloc[idx_val].reset_index(drop=True)

    # Sicherheitscheck: keine Gruppenüberschneidung
    for a, b, name in [
        (df_train, df_val, "train/val"),
        (df_train, df_test, "train/test"),
        (df_val, df_test, "val/test"),
    ]:
        ga = set(a["group_id"].unique())
        gb = set(b["group_id"].unique())
        assert ga.isdisjoint(gb), f"Group leakage zwischen {name}"

    return df_train, df_val, df_test


def to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_epoch(model, loader, optim, device, loss_fn):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optim.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, loss_fn):
    model.eval()
    total = 0.0
    n = 0
    preds = []
    trues = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        out = model(xb)
        loss = loss_fn(out, yb)
        total += loss.item() * xb.size(0)
        n += xb.size(0)
        preds.append(out.cpu().numpy())
        trues.append(yb.cpu().numpy())
    if preds:
        preds = np.vstack(preds)
        trues = np.vstack(trues)
    else:
        preds = np.empty((0, 3))
        trues = np.empty((0, 3))
    return total / max(n, 1), preds, trues


def main():
    ap = argparse.ArgumentParser(description="Multi-Output PyTorch – Unseen Test (gruppierter Split)")
    ap.add_argument("--dataset", required=True, help="Pfad zur CSV (z. B. results/datasets/merged_sigmos_wavlm_multiwer.csv)")
    ap.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis")
    ap.add_argument("--use_snr", action="store_true", help="SNR-Spalte als Feature verwenden")
    ap.add_argument("--test_size", type=float, default=0.1, help="Anteil Test")
    ap.add_argument("--val_size", type=float, default=0.1, help="Anteil Val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== Unseen-Evaluation (gruppiert) auf {args.dataset} ===")

    # Repro
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Daten laden
    df = pd.read_csv(args.dataset)
    if "filename" not in df.columns:
        raise RuntimeError("Spalte 'filename' wird benötigt.")
    for t in TARGETS:
        if t not in df.columns:
            raise RuntimeError(f"Spalte '{t}' fehlt.")

    # Gruppen-ID
    df["group_id"] = df["filename"].map(stem_from_filename)

    # Features/Targets
    X_all, y_all = build_xy(df, use_snr=args.use_snr)

    # Split
    df_train, df_val, df_test = split_grouped(df, test_size=args.test_size, val_size=args.val_size, random_state=args.seed)

    # Indizes auf X/y abbilden
    idx_map = {i: i for i in range(len(df))}
    train_idx = df_train.index.values
    val_idx = df_val.index.values
    test_idx = df_test.index.values

    X = X_all.values
    y = y_all.values

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Skalierung
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Dataloaders
    train_loader = to_loader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = to_loader(X_val, y_val, batch_size=args.batch_size, shuffle=False)
    test_loader = to_loader(X_test, y_test, batch_size=args.batch_size, shuffle=False)

    # Modell
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = MLP(in_dim=X_train.shape[1], out_dim=len(TARGETS), hidden=[512, 256, 128], p_drop=0.15).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    # Training mit Early Stopping
    best_val = float("inf")
    best_state = None
    patience = 20
    bad = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, device, loss_fn)
        val_loss, _, _ = eval_epoch(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    # Bestes Modell laden
    if best_state is not None:
        model.load_state_dict(best_state)

    # Auswertung Val/Test
    _, pred_val, true_val = eval_epoch(model, val_loader, device, loss_fn)
    _, pred_test, true_test = eval_epoch(model, test_loader, device, loss_fn)

    # Metriken pro Target
    rows = []
    for i, tgt in enumerate(TARGETS):
        r2_v, rmse_v, mae_v = r2_rmse_mae(true_val[:, i], pred_val[:, i])
        r2_t, rmse_t, mae_t = r2_rmse_mae(true_test[:, i], pred_test[:, i])
        rows.append({
            "target": tgt,
            "split": "val",
            "r2": r2_v, "rmse": rmse_v, "mae": mae_v
        })
        rows.append({
            "target": tgt,
            "split": "test",
            "r2": r2_t, "rmse": rmse_t, "mae": mae_t
        })
    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(args.out_dir, "unseen_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Test-Predictions speichern (inkl. filename, snr falls vorhanden)
    out_pred = df_test[["filename"] + ([ "snr"] if "snr" in df_test.columns else [])].copy()
    for i, tgt in enumerate(TARGETS):
        out_pred[f"{tgt}_true"] = true_test[:, i]
        out_pred[f"{tgt}_pred"] = pred_test[:, i]
    pred_path = os.path.join(args.out_dir, "unseen_predictions.csv")
    out_pred.to_csv(pred_path, index=False)

    print(f"Ergebnisse gespeichert unter:\n- {metrics_path}\n- {pred_path}")


if __name__ == "__main__":
    main()