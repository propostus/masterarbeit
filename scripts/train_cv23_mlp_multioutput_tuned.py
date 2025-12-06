# scripts/train_cv23_mlp_multioutput_tuned.py
# -----------------------------------------------------------------------------
# Multi-Output-MLP mit Hyperparameter-Tuning für CV23-balanced
# - Input:  merged_sigmos_wavlm_cv23_balanced_multiwer.csv
#           + optionale handcrafted Audiofeatures
# - Targets: wer_tiny, wer_base, wer_small
# - Split: gruppiert nach client_id (Train / Val / Test)
# - Features: alle numerischen Spalten außer IDs/Meta/Targets
# - Skalierung: StandardScaler (auf Train fitten, auf Val/Test anwenden)
# - Tuning-Kriterium: mittlerer CCC über alle drei WER-Ziele (Validation)
# - Ausgabe:
#   * bestes Modell: <out_dir>/mlp_multioutput_best.pt
#   * Scaler:        <out_dir>/scaler.pkl
#   * Tuning-Tabelle:<out_dir>/mlp_hparam_search_results.csv
# -----------------------------------------------------------------------------

import os
import argparse
import json
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def concordance_correlation_coefficient(y_true, y_pred) -> float:
    """
    CCC gemäß Lin (1989).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)
    return float(ccc)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Berechnet r2, rmse, mae, ccc für einen Zielvektor.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae, "ccc": ccc}


# --------------------------------------------------------------------------
# Dataset / DataLoader
# --------------------------------------------------------------------------

class WerDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_dataloader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    ds = WerDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------
# Modell
# --------------------------------------------------------------------------

class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int = 3, dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------
# Daten laden und vorbereiten
# --------------------------------------------------------------------------

def load_and_prepare_data(
    dataset_path: str,
    extra_features_csv: str | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Lädt den großen CV23-Merged-Datensatz und merged optional zusätzliche
    Audiofeatures. Gibt das volle DataFrame + Feature-Spaltennamen (ohne Targets)
    zurück.
    """
    print(f"Lade Datensatz von: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)

    if extra_features_csv is not None:
        print(f"Lade zusätzliche Features von: {extra_features_csv}")
        df_extra = pd.read_csv(extra_features_csv, low_memory=False)
        # Spalte normalisieren: file/filename
        col_name = "filename" if "filename" in df_extra.columns else "file"
        df_extra[col_name] = df_extra[col_name].astype(str).str.lower()
        df["filename"] = df["filename"].astype(str).str.lower()
        df = df.merge(df_extra, how="left", on=col_name)
        print(f"Nach Merge mit Extra-Features: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Targets
    targets = ["wer_tiny", "wer_base", "wer_small"]
    df = df.dropna(subset=targets)
    print(f"Nach Drop von NaN-Targets: {df.shape[0]} Zeilen")

    # Feature-Spalten bestimmen
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
        c
        for c in df.columns
        if (c not in drop_cols) and np.issubdtype(df[c].dtype, np.number)
    ]

    print(f"Anzahl Feature-Spalten: {len(feature_cols)}")
    return df, feature_cols


def train_val_test_split_by_speaker(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Gruppierter Split nach client_id: erst Train+Val vs. Test, dann Train vs. Val.
    test_size/val_size sind bezogen auf den Gesamtdatensatz (ungefähr).
    """
    groups = df["client_id"].astype(str)

    # 1) Train+Val vs Test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # 2) Train vs Val innerhalb Train+Val
    groups_trainval = trainval_df["client_id"].astype(str)
    relative_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=groups_trainval))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    print(
        f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}"
    )
    print(
        f"Unique Speaker (Train/Val/Test): "
        f"{train_df['client_id'].nunique()} / {val_df['client_id'].nunique()} / {test_df['client_id'].nunique()}"
    )

    return train_df, val_df, test_df


def prepare_numpy_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrahiert X/y für Train/Val/Test als NumPy-Arrays.
    """
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[["wer_tiny", "wer_base", "wer_small"]].to_numpy(dtype=np.float32)

    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df[["wer_tiny", "wer_base", "wer_small"]].to_numpy(dtype=np.float32)

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df[["wer_tiny", "wer_base", "wer_small"]].to_numpy(dtype=np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test


# --------------------------------------------------------------------------
# Training für eine Hyperparameter-Konfiguration
# --------------------------------------------------------------------------

def train_one_config(
    config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Trainiert das MLP mit gegebener Konfiguration und gibt zwei Dicts zurück:
      - metrics: Aggregierte Val/Test-Metriken + Mittelwerte
      - per_target: Metriken pro Zielvariable (für späteren CSV-Export optional)
    """

    batch_size = config["batch_size"]
    hidden_sizes = config["hidden_sizes"]
    dropout = config["dropout"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch == 1) or (epoch % 10 == 0):
            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        if epochs_no_improve >= patience:
            print(f"Early Stopping nach {epoch} Epochen (Val-Loss verbessert sich nicht mehr).")
            break

    # Bestes Modell laden
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Val- und Test-Metriken berechnen
    def predict(loader: DataLoader) -> np.ndarray:
        model.eval()
        preds_list = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                preds_list.append(preds)
        return np.vstack(preds_list)

    y_val_pred = predict(val_loader)
    y_test_pred = predict(test_loader)

    targets = ["wer_tiny", "wer_base", "wer_small"]
    val_metrics_per_target = []
    test_metrics_per_target = []

    for i, tname in enumerate(targets):
        vm = evaluate_regression(y_val[:, i], y_val_pred[:, i])
        tm = evaluate_regression(y_test[:, i], y_test_pred[:, i])
        vm["target"] = tname
        tm["target"] = tname
        val_metrics_per_target.append(vm)
        test_metrics_per_target.append(tm)

    # Mittelwerte bilden
    val_ccc_mean = float(np.mean([m["ccc"] for m in val_metrics_per_target]))
    test_ccc_mean = float(np.mean([m["ccc"] for m in test_metrics_per_target]))
    val_r2_mean = float(np.mean([m["r2"] for m in val_metrics_per_target]))
    test_r2_mean = float(np.mean([m["r2"] for m in test_metrics_per_target]))

    metrics = {
        "config": json.dumps(config),
        "val_mean_ccc": val_ccc_mean,
        "test_mean_ccc": test_ccc_mean,
        "val_mean_r2": val_r2_mean,
        "test_mean_r2": test_r2_mean,
        "val_loss": best_val_loss,
    }

    # Wir hängen pro Target noch Namen an (optional für CSV-Auswertung)
    per_target = {
        "val": val_metrics_per_target,
        "test": test_metrics_per_target,
    }

    return metrics, per_target, best_state_dict


# --------------------------------------------------------------------------
# Hyperparameter-Suche
# --------------------------------------------------------------------------

def build_config_grid() -> List[Dict[str, Any]]:
    """
    Baut ein Grid an sinnvollen Konfigurationen.
    Hier relativ ausführlich, aber nicht komplett explodierend.
    """
    hidden_options = [
        [512, 256],
        [512, 256, 128],
        [768, 384],
        [768, 384, 192],
    ]
    dropout_options = [0.1, 0.2, 0.3]
    lr_options = [1e-3, 5e-4]
    batch_options = [256, 512, 1024]
    weight_decay_options = [1e-4, 5e-5]

    configs: List[Dict[str, Any]] = []
    for h in hidden_options:
        for d in dropout_options:
            for lr in lr_options:
                for bs in batch_options:
                    for wd in weight_decay_options:
                        cfg = {
                            "hidden_sizes": h,
                            "dropout": d,
                            "lr": lr,
                            "batch_size": bs,
                            "weight_decay": wd,
                        }
                        configs.append(cfg)
    return configs


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter-Tuning für Multi-Output-MLP auf CV23-balanced"
    )
    parser.add_argument("--dataset", required=True, help="Pfad zur merged_sigmos_wavlm CSV")
    parser.add_argument("--extra_features_csv", type=str, default=None,
                        help="Pfad zu zusätzlichen Audiofeatures (optional)")
    parser.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | mps | auto")
    parser.add_argument("--epochs", type=int, default=60, help="Maximale Epochen pro Config")
    parser.add_argument("--patience", type=int, default=10, help="Early-Stopping-Patience")
    parser.add_argument("--test_size", type=float, default=0.2, help="Testanteil")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validierungsanteil")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--max_configs", type=int, default=24,
                        help="Maximale Anzahl getesteter Hyperparameter-Konfigurationen")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

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

    # Daten laden
    df, feature_cols = load_and_prepare_data(args.dataset, args.extra_features_csv)

    # Split nach Speaker
    train_df, val_df, test_df = train_val_test_split_by_speaker(
        df, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )

    # NumPy-Matrizen
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_numpy_matrices(
        train_df, val_df, test_df, feature_cols
    )

    # Skalierung (auf Train fitten)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Scaler speichern (für spätere Nutzung des besten Modells)
    scaler_path = os.path.join(args.out_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler gespeichert unter: {scaler_path}")

    # Hyperparameter-Grid
    all_configs = build_config_grid()
    print(f"Anzahl möglicher Konfigurationen im Grid: {len(all_configs)}")
    if args.max_configs is not None and args.max_configs < len(all_configs):
        # deterministisch subsetten
        all_configs = all_configs[: args.max_configs]
        print(f"Beschränke auf die ersten {len(all_configs)} Konfigurationen (max_configs).")

    results_rows: List[Dict[str, Any]] = []
    best_global: Dict[str, Any] | None = None
    best_state_dict = None

    for idx, cfg in enumerate(all_configs, start=1):
        print("\n" + "=" * 80)
        print(f"Konfiguration {idx}/{len(all_configs)}: {cfg}")
        print("=" * 80)

        metrics, per_target, state_dict = train_one_config(
            cfg,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            device=device,
            max_epochs=args.epochs,
            patience=args.patience,
        )

        # Zeile für CSV vorbereiten
        row = {
            "config_idx": idx,
            "hidden_sizes": str(cfg["hidden_sizes"]),
            "dropout": cfg["dropout"],
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "weight_decay": cfg["weight_decay"],
            "val_mean_ccc": metrics["val_mean_ccc"],
            "test_mean_ccc": metrics["test_mean_ccc"],
            "val_mean_r2": metrics["val_mean_r2"],
            "test_mean_r2": metrics["test_mean_r2"],
            "val_loss": metrics["val_loss"],
        }

        # optional: pro Target die CCC-Werte mit aufnehmen
        for split_name in ["val", "test"]:
            for m in per_target[split_name]:
                prefix = f"{split_name}_{m['target']}"
                row[f"{prefix}_r2"] = m["r2"]
                row[f"{prefix}_rmse"] = m["rmse"]
                row[f"{prefix}_mae"] = m["mae"]
                row[f"{prefix}_ccc"] = m["ccc"]

        results_rows.append(row)

        # global best anhand val_mean_ccc
        if (best_global is None) or (metrics["val_mean_ccc"] > best_global["val_mean_ccc"]):
            best_global = metrics | {"config_idx": idx, "config": cfg}
            best_state_dict = state_dict
            print(f"==> Neue beste Konfiguration (nach Val-Mean-CCC): {best_global}")

    # Ergebnisse speichern
    results_df = pd.DataFrame(results_rows)
    results_csv_path = os.path.join(args.out_dir, "mlp_hparam_search_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nTuning-Ergebnisse gespeichert unter: {results_csv_path}")

    if best_global is not None and best_state_dict is not None:
        best_model_path = os.path.join(args.out_dir, "mlp_multioutput_best.pt")
        # Modell-Architektur entsprechend der besten Config rekonstruieren
        best_cfg = best_global["config"]
        if isinstance(best_cfg, dict):
            cfg = best_cfg
        else:
            cfg = json.loads(best_cfg)

        model_best = MultiOutputMLP(
            input_dim=X_train.shape[1],
            hidden_sizes=cfg["hidden_sizes"],
            output_dim=3,
            dropout=cfg["dropout"],
        )
        model_best.load_state_dict(best_state_dict)
        torch.save(
            {
                "state_dict": model_best.state_dict(),
                "config": cfg,
                "input_dim": X_train.shape[1],
                "output_dim": 3,
            },
            best_model_path,
        )
        print(f"Bestes Modell gespeichert unter: {best_model_path}")
        print("Beste Konfiguration (nach Val-Mean-CCC):")
        print(best_global)
    else:
        print("Keine erfolgreiche Konfiguration gefunden.")


if __name__ == "__main__":
    main()