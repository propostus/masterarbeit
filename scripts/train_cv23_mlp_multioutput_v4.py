import os
import json
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# ------------------------------------------------------
# Multi-Output MLP
# ------------------------------------------------------
class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim=3, dropout=0.2):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------
# DataLoader-Helfer
# ------------------------------------------------------
def make_loaders(
    X_train, y_train,
    X_val, y_val,
    batch_size,
    X_test=None, y_test=None
):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_loader = None
    if X_test is not None and y_test is not None:
        test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ------------------------------------------------------
# Evaluation auf einem Loader
# ------------------------------------------------------
def eval_on_loader(model, loader, device):
    model.eval()
    preds_all = []
    trues_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.numpy()
            preds = model(xb).cpu().numpy()
            preds_all.append(preds)
            trues_all.append(yb)

    preds_all = np.vstack(preds_all)
    trues_all = np.vstack(trues_all)

    r2_tiny = r2_score(trues_all[:, 0], preds_all[:, 0])
    r2_base = r2_score(trues_all[:, 1], preds_all[:, 1])
    r2_small = r2_score(trues_all[:, 2], preds_all[:, 2])
    mean_r2 = float(np.mean([r2_tiny, r2_base, r2_small]))

    metrics = {
        "r2_tiny": r2_tiny,
        "r2_base": r2_base,
        "r2_small": r2_small,
        "r2_mean": mean_r2,
    }
    return mean_r2, metrics


# ------------------------------------------------------
# Training einer Konfiguration
# ------------------------------------------------------
def train_one_config(
    config,
    device,
    X_train, y_train,
    X_val, y_val,
    max_epochs,
    patience
):
    cfg_id = config["config_id"]
    hidden_sizes = config["hidden_sizes"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    dropout = config["dropout"]
    weight_decay = config["weight_decay"]

    print(f"\n===== Config {cfg_id}/{config['total_configs']} =====")
    print(f"Konfiguration: {config}")

    input_dim = X_train.shape[1]
    model = MultiOutputMLP(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        output_dim=3,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )
    loss_fn = nn.MSELoss()

    train_loader, val_loader, _ = make_loaders(
        X_train, y_train, X_val, y_val, batch_size
    )

    best_state = None
    best_val_r2 = -1e9
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        model.eval()
        val_r2_mean, _ = eval_on_loader(model, val_loader, device)
        scheduler.step(val_r2_mean)

        if val_r2_mean > best_val_r2 + 1e-4:
            best_val_r2 = val_r2_mean
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    print(f"Beste Val-R² (mean) für Config {cfg_id}: {best_val_r2:.4f}")
    return best_val_r2, best_state


# ------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Multi-Output MLP on CV23 (SigMOS+WavLM)")
    parser.add_argument("--dataset", required=True, help="CSV mit Features + Targets")
    parser.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps | auto")
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # Device-Auswahl
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
    os.makedirs(args.out_dir, exist_ok=True)

    # Daten laden
    df = pd.read_csv(args.dataset)
    print("Shape:", df.shape)

    # Targets
    target_cols = ["wer_tiny", "wer_base", "wer_small"]

    # Reihen mit NaN in Targets verwerfen
    df = df.dropna(subset=target_cols)
    print("Nach Drop von NaN-Targets:", df.shape[0], "Zeilen")

    # Feature-Spalten:
    # - nur numerische Spalten
    # - keine Targets
    # - keine Meta-Spalten (ID / Text)
    exclude_cols = set([
        "filename",
        "client_id",
        "age",
        "gender",
        "sentence",
        "wer_tiny",
        "wer_base",
        "wer_small",
    ])

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    print("Anzahl Feature-Spalten:", len(feature_cols))

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    # Speaker-IDs
    if "client_id" not in df.columns:
        raise RuntimeError("Spalte 'client_id' wird für den speaker-basierten Split benötigt.")
    speaker_ids = df["client_id"].astype(str).values

    # Speaker-basierten Split definieren
    rng = np.random.default_rng(args.random_state)
    unique_speakers = np.unique(speaker_ids)
    rng.shuffle(unique_speakers)

    n_total_spk = len(unique_speakers)
    n_train_spk = int(0.7 * n_total_spk)
    n_val_spk = int(0.15 * n_total_spk)

    train_spk = unique_speakers[:n_train_spk]
    val_spk = unique_speakers[n_train_spk:n_train_spk + n_val_spk]
    test_spk = unique_speakers[n_train_spk + n_val_spk:]

    mask_train = np.isin(speaker_ids, train_spk)
    mask_val = np.isin(speaker_ids, val_spk)
    mask_test = np.isin(speaker_ids, test_spk)

    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Unique Speaker (Train/Val/Test): "
          f"{len(np.unique(speaker_ids[mask_train]))} / "
          f"{len(np.unique(speaker_ids[mask_val]))} / "
          f"{len(np.unique(speaker_ids[mask_test]))}")

    # Skalierung
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Feature-Infos speichern
    with open(os.path.join(args.out_dir, "feature_cols.txt"), "w") as f:
        f.write("\n".join(feature_cols))
    torch.save(scaler, os.path.join(args.out_dir, "scaler.pkl"))

    # Hyperparameter-Grid
    grid = []
    hidden_sets = [
        [512, 256, 128],
    ]
    learning_rates = [0.001, 0.0005]
    batch_sizes = [256, 512]
    dropouts = [0.2, 0.3]
    weight_decays = [0.0, 0.0001]

    cfg_id = 1
    total_cfg = len(hidden_sets) * len(learning_rates) * len(batch_sizes) * len(dropouts) * len(weight_decays)
    for h in hidden_sets:
        for lr in learning_rates:
            for bs in batch_sizes:
                for d in dropouts:
                    for wd in weight_decays:
                        grid.append({
                            "config_id": cfg_id,
                            "total_configs": total_cfg,
                            "hidden_sizes": h,
                            "lr": lr,
                            "batch_size": bs,
                            "dropout": d,
                            "weight_decay": wd,
                        })
                        cfg_id += 1

    best_val_r2 = -1e9
    best_config = None
    best_state = None
    all_results = []

    # Grid Search
    for cfg in grid:
        val_r2_mean, state_dict = train_one_config(
            cfg, device,
            X_train_s, y_train,
            X_val_s, y_val,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        all_results.append({
            "config_id": cfg["config_id"],
            "hidden_sizes": str(cfg["hidden_sizes"]),
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "dropout": cfg["dropout"],
            "weight_decay": cfg["weight_decay"],
            "val_r2_mean": val_r2_mean,
        })
        if val_r2_mean > best_val_r2:
            best_val_r2 = val_r2_mean
            best_config = cfg
            best_state = state_dict

    # Grid-Search-Resultate speichern
    df_gs = pd.DataFrame(all_results)
    df_gs.to_csv(os.path.join(args.out_dir, "mlp_v4_hparam_search_results.csv"), index=False)

    print("\n===== Beste Konfiguration (Val-R²) =====")
    print(best_config)
    print(f"Val-R² (mean): {best_val_r2:.4f}")

    # Bestes Modell initialisieren und State laden
    model = MultiOutputMLP(
        input_dim=X_train_s.shape[1],
        hidden_sizes=best_config["hidden_sizes"],
        output_dim=3,
        dropout=best_config["dropout"],
    ).to(device)
    model.load_state_dict(best_state)

    # Test-Evaluation mit dem besten Val-Modell (ohne Retraining)
    _, test_loader = None, None
    _, _, test_loader = make_loaders(
        X_train_s, y_train, X_val_s, y_val,
        batch_size=best_config["batch_size"],
        X_test=X_test_s, y_test=y_test,
    )
    test_mean_r2, test_metrics = eval_on_loader(model, test_loader, device)

    # Modell und Metriken speichern
    torch.save(model.state_dict(), os.path.join(args.out_dir, "mlp_v4_best_model.pt"))
    with open(os.path.join(args.out_dir, "mlp_v4_best_config.json"), "w") as f:
        json.dump(
            {
                "best_config": best_config,
                "best_val_r2_mean": best_val_r2,
                "test_r2_mean": test_mean_r2,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )
    pd.DataFrame([test_metrics]).to_csv(
        os.path.join(args.out_dir, "mlp_v4_test_metrics.csv"),
        index=False,
    )

    print("\nFertig.")
    print("Test-R² (mean):", test_mean_r2)
    print("Test-Metriken pro Target:", test_metrics)
    print("Ausgaben gespeichert in:", args.out_dir)


if __name__ == "__main__":
    main()