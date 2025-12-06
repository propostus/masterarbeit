# scripts/train_cv23_speakerlevel_full.py
# Vollständige Trainings-Pipeline für CV23-balanced:
# - Clip-Level-Dataset laden (SigMOS + WavLM + WER + Meta)
# - Optional: Handcrafted Audiofeatures mergen
# - Speaker-Level-Features (mean/std je client_id) berechnen und zurück mergen
# - Gruppenbasierter Train/Test-Split nach client_id
# - Training: LightGBM & CatBoost (pro WER-Target)
# - Training: Multi-Output-MLP mit Hyperparameter-Suche und Early Stopping
# - Speicherung aller Metriken und der besten Modelle
#
# Hinweis:
# - Erwartet, dass das Eingabedataset eine Spalte "client_id" und die Targets
#   "wer_tiny", "wer_base", "wer_small" enthält.
# - WavLM-Embeddings werden als numerische Spalten mit nur Ziffern im Namen angenommen.
# - Optionales extra_features_csv sollte pro Clip Features mit Spalte "filename" enthalten.

import os
import argparse
import json
import gc
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------------------

def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance Correlation Coefficient (CCC)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-12)
    return float(ccc)


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


# -------------------------------------------------------------------
# Datenvorbereitung
# -------------------------------------------------------------------

def load_and_prepare_dataset(dataset_path, extra_features_csv=None, max_rows=None):
    """
    Lädt das Clip-Level-Dataset, merged optional zusätzliche Features
    und baut Speaker-Level-Features (mean/std je client_id), die wieder
    auf Clip-Level zurückgemerged werden.
    """
    print_section("Lade Clip-Level-Dataset")
    print(f"Lese: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False, nrows=max_rows)
    print(f"Geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    required_cols = ["filename", "client_id", "wer_tiny", "wer_base", "wer_small"]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"Erwartete Spalte '{col}' nicht in Dataset vorhanden.")

    # Optional: zusätzliche handcrafted Features mergen
    if extra_features_csv is not None and os.path.exists(extra_features_csv):
        print_section("Mergen: Handcrafted Audiofeatures")
        print(f"Lese: {extra_features_csv}")
        df_extra = pd.read_csv(extra_features_csv, low_memory=False)
        if "filename" not in df_extra.columns:
            raise RuntimeError("extra_features_csv muss eine Spalte 'filename' enthalten.")
        # Nur numerische Spalten zusätzlich mergen
        extra_num_cols = [c for c in df_extra.columns if c != "filename" and
                          np.issubdtype(df_extra[c].dtype, np.number)]
        keep_cols = ["filename"] + extra_num_cols
        df_extra = df_extra[keep_cols]
        before = df.shape[1]
        df = df.merge(df_extra, on="filename", how="left")
        after = df.shape[1]
        print(f"Handcrafted-Features gemerged: +{after - before} Spalten")

    # Numerische Spalten identifizieren
    target_cols = ["wer_tiny", "wer_base", "wer_small"]
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    feature_num_cols = [c for c in numeric_cols if c not in target_cols]

    print_section("Speaker-Level-Aggregation (mean/std je client_id)")
    print(f"Anzahl numerischer Feature-Spalten (ohne Targets): {len(feature_num_cols)}")

    # Gruppieren nach client_id und mean/std berechnen
    t0 = time()
    grouped = df.groupby("client_id")[feature_num_cols].agg(["mean", "std"])
    # MultiIndex-Spalten flatten
    new_cols = []
    for base, stat in grouped.columns:
        new_cols.append(f"{base}_spk_{stat}")
    grouped.columns = new_cols

    # Anzahl Clips pro Sprecher
    counts = df.groupby("client_id")["filename"].count().rename("spk_num_clips")

    speaker_stats = pd.concat([grouped, counts], axis=1)
    print(f"Speaker-Stats: {speaker_stats.shape[0]} Sprecher, {speaker_stats.shape[1]} Spalten")
    print(f"Aggregation fertig in {time() - t0:.1f} s")

    # Zurück auf Clip-Level mergen
    t0 = time()
    df = df.merge(speaker_stats, left_on="client_id", right_index=True, how="left")
    print(f"Nach Merge Speaker-Stats: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    print(f"Merge fertig in {time() - t0:.1f} s")

    # Alle numerischen Features erneut bestimmen und auf float32 casten
    numeric_cols_final = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    for c in numeric_cols_final:
        df[c] = df[c].astype(np.float32)

    return df, target_cols


def build_matrices(df, target_cols):
    """
    Baut X, y, groups und gibt auch die Feature-Namen zurück.
    """
    print_section("Baue Feature-Matrizen")
    # Gruppen (client_id) für den Split
    groups = df["client_id"].values

    # Numerische Spalten, Targets ausschließen
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    feature_cols = [c for c in numeric_cols if c not in target_cols]

    print(f"Anzahl Feature-Spalten: {len(feature_cols)}")
    print(f"Targets: {target_cols}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    print(f"X-Shape: {X.shape}, y-Shape: {y.shape}")
    return X, y, groups, feature_cols


# -------------------------------------------------------------------
# Tree-Modelle: LightGBM & CatBoost
# -------------------------------------------------------------------

def train_tree_models(X_train, y_train, X_test, y_test, target_cols, out_dir):
    results = []

    for idx, target in enumerate(target_cols):
        print_section(f"Training LightGBM & CatBoost für {target}")
        y_tr = y_train[:, idx]
        y_te = y_test[:, idx]

        # LightGBM
        try:
            print("LightGBM Training ...")
            lgb = LGBMRegressor(
                n_estimators=700,
                learning_rate=0.03,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=-1,
                random_state=42,
            )
            lgb.fit(X_train, y_tr)
            preds = lgb.predict(X_test)
            r2 = r2_score(y_te, preds)
            rmse = mean_squared_error(y_te, preds, squared=False)
            mae = mean_absolute_error(y_te, preds)
            ccc = concordance_correlation_coefficient(y_te, preds)

            results.append({
                "model": "LightGBM",
                "target": target,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "ccc": ccc,
            })

            model_path = os.path.join(out_dir, f"lightgbm_{target}.pkl")
            import joblib
            joblib.dump(lgb, model_path)
            print(f"LightGBM für {target} gespeichert unter: {model_path}")

        except Exception as e:
            print(f"Fehler beim LightGBM-Training für {target}: {e}")

        # CatBoost
        try:
            print("CatBoost Training ...")
            cat = CatBoostRegressor(
                iterations=700,
                learning_rate=0.03,
                depth=8,
                loss_function="RMSE",
                verbose=False,
                random_seed=42,
            )
            cat.fit(X_train, y_tr)
            preds = cat.predict(X_test)
            r2 = r2_score(y_te, preds)
            rmse = mean_squared_error(y_te, preds, squared=False)
            mae = mean_absolute_error(y_te, preds)
            ccc = concordance_correlation_coefficient(y_te, preds)

            results.append({
                "model": "CatBoost",
                "target": target,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "ccc": ccc,
            })

            model_path = os.path.join(out_dir, f"catboost_{target}.cbm")
            cat.save_model(model_path)
            print(f"CatBoost für {target} gespeichert unter: {model_path}")

        except Exception as e:
            print(f"Fehler beim CatBoost-Training für {target}: {e}")

    return results


# -------------------------------------------------------------------
# MLP Multi-Output
# -------------------------------------------------------------------

class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout, output_dim=3):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp_multioutput(
    X_train,
    y_train,
    X_test,
    y_test,
    groups_train,
    target_cols,
    out_dir,
    device=None,
):
    print_section("Training Multi-Output-MLP mit Hyperparameter-Suche")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Verwende Gerät für MLP: {device}")

    # Train/Val-Split innerhalb des Trainingssets (gruppenbasiert)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=123)
    tr_idx, val_idx = next(gss.split(X_train, y_train, groups_train))
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # DataLoader
    def make_loader(X, y, batch_size, shuffle):
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    configs = [
        {"hidden_sizes": [512, 256], "dropout": 0.2, "lr": 1e-3, "batch_size": 1024},
        {"hidden_sizes": [1024, 512], "dropout": 0.2, "lr": 1e-3, "batch_size": 1024},
        {"hidden_sizes": [512, 256, 128], "dropout": 0.3, "lr": 5e-4, "batch_size": 1024},
        {"hidden_sizes": [1024, 512, 256], "dropout": 0.3, "lr": 5e-4, "batch_size": 1024},
    ]

    max_epochs = 40
    patience = 6

    best_val_r2 = -1e9
    best_model_state = None
    best_config = None
    best_test_metrics = None

    for cfg_idx, cfg in enumerate(configs):
        print(f"\nKonfiguration {cfg_idx + 1}/{len(configs)}: {cfg}")
        model = MultiOutputMLP(
            input_dim=input_dim,
            hidden_sizes=cfg["hidden_sizes"],
            dropout=cfg["dropout"],
            output_dim=output_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        criterion = nn.MSELoss()

        train_loader = make_loader(X_tr, y_tr, cfg["batch_size"], shuffle=True)
        val_loader = make_loader(X_val, y_val, cfg["batch_size"], shuffle=False)

        best_cfg_val_r2 = -1e9
        best_cfg_state = None
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            with torch.no_grad():
                all_preds = []
                all_true = []
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pr = model(xb)
                    all_preds.append(pr.cpu().numpy())
                    all_true.append(yb.cpu().numpy())
                y_val_pred = np.vstack(all_preds)
                y_val_true = np.vstack(all_true)

            # mittleres R² über alle Targets
            r2_targets = []
            for t_idx in range(output_dim):
                r2_targets.append(r2_score(y_val_true[:, t_idx], y_val_pred[:, t_idx]))
            mean_r2 = float(np.mean(r2_targets))

            print(f"Epoch {epoch:03d} | TrainLoss={train_loss:.4f} | Val R2 mean={mean_r2:.4f}")

            if mean_r2 > best_cfg_val_r2 + 1e-4:
                best_cfg_val_r2 = mean_r2
                epochs_no_improve = 0
                best_cfg_state = deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early Stopping für diese Konfiguration.")
                    break

        # Test-Performance mit bestem Zustand der Konfiguration
        if best_cfg_state is None:
            print("Keine Verbesserung in dieser Konfiguration, überspringe Auswertung.")
            continue

        model.load_state_dict(best_cfg_state)

        # Testdaten
        test_loader = make_loader(X_test, y_test, cfg["batch_size"], shuffle=False)
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_true = []
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pr = model(xb)
                all_preds.append(pr.cpu().numpy())
                all_true.append(yb.cpu().numpy())
        y_test_pred = np.vstack(all_preds)
        y_test_true = np.vstack(all_true)

        test_metrics = []
        for t_idx, target in enumerate(target_cols):
            r2 = r2_score(y_test_true[:, t_idx], y_test_pred[:, t_idx])
            rmse = mean_squared_error(y_test_true[:, t_idx], y_test_pred[:, t_idx], squared=False)
            mae = mean_absolute_error(y_test_true[:, t_idx], y_test_pred[:, t_idx])
            ccc = concordance_correlation_coefficient(y_test_true[:, t_idx], y_test_pred[:, t_idx])
            test_metrics.append({
                "model": "MLP",
                "target": target,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "ccc": ccc,
                "config": str(cfg),
            })

        mean_r2_test = float(np.mean([m["r2"] for m in test_metrics]))
        print(f"Konfiguration {cfg_idx + 1}: mittleres Test-R2 = {mean_r2_test:.4f}")

        # Globale Bestwahl nach Validierungs-R2
        if best_cfg_val_r2 > best_val_r2:
            best_val_r2 = best_cfg_val_r2
            best_model_state = deepcopy(best_cfg_state)
            best_config = cfg
            best_test_metrics = test_metrics

    # Bestes Modell speichern
    mlp_results = []
    if best_model_state is not None:
        model = MultiOutputMLP(
            input_dim=input_dim,
            hidden_sizes=best_config["hidden_sizes"],
            dropout=best_config["dropout"],
            output_dim=output_dim,
        )
        model.load_state_dict(best_model_state)
        model_path = os.path.join(out_dir, "mlp_multioutput_best.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Bestes MLP-Modell gespeichert unter: {model_path}")
        print(f"Beste Konfiguration: {best_config}")
        mlp_results.extend(best_test_metrics)

        # Konfiguration zusätzlich als JSON speichern
        cfg_path = os.path.join(out_dir, "mlp_best_config.json")
        with open(cfg_path, "w") as f:
            json.dump({
                "best_config": best_config,
                "best_val_r2": best_val_r2,
            }, f, indent=2)
        print(f"MLP-Konfiguration gespeichert unter: {cfg_path}")
    else:
        print("MLP-Training lieferte kein gültiges Modell.")

    return mlp_results


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CV23 Speaker-Level Full Pipeline")
    parser.add_argument("--dataset", required=True, help="Pfad zur Clip-Level CSV (merged_sigmos_wavlm_cv23_balanced_multiwer.csv)")
    parser.add_argument("--extra_features_csv", type=str, default=None,
                        help="Optional: CSV mit zusätzlichen Clip-Level-Features (Spalte 'filename')")
    parser.add_argument("--out_dir", required=True, help="Ausgabeverzeichnis für Modelle und Metriken")
    parser.add_argument("--test_size", type=float, default=0.2, help="Testanteil für GroupShuffleSplit")
    parser.add_argument("--random_state", type=int, default=42, help="Random State für GroupShuffleSplit")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Optional: maximale Zeilenanzahl (Debugging)")

    parser.add_argument("--disable_mlp", action="store_true",
                        help="Wenn gesetzt, wird kein MLP trainiert (nur Tree-Modelle).")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Dataset laden und Speaker-Stats mergen
    df, target_cols = load_and_prepare_dataset(
        dataset_path=args.dataset,
        extra_features_csv=args.extra_features_csv,
        max_rows=args.max_rows,
    )

    # 2) Matrizen bauen
    X, y, groups, feature_cols = build_matrices(df, target_cols)

    # Speicher etwas freigeben
    del df
    gc.collect()

    # 3) Gruppenbasierter Train/Test-Split
    print_section("Gruppenbasierter Train/Test-Split nach client_id")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    print(f"Train: {X_train.shape[0]} Zeilen, Test: {X_test.shape[0]} Zeilen")
    print(f"Unique Sprecher (Train/Test): {len(np.unique(groups_train))} / {len(np.unique(groups_test))}")

    # 4) Tree-Modelle trainieren
    all_results = []
    try:
        tree_results = train_tree_models(X_train, y_train, X_test, y_test, target_cols, args.out_dir)
        all_results.extend(tree_results)
    except Exception as e:
        print(f"Fehler im Tree-Training-Block: {e}")

    # 5) MLP Multi-Output trainieren
    if not args.disable_mlp:
        try:
            mlp_results = train_mlp_multioutput(
                X_train, y_train, X_test, y_test, groups_train, target_cols, args.out_dir
            )
            all_results.extend(mlp_results)
        except Exception as e:
            print(f"Fehler im MLP-Block: {e}")
    else:
        print("MLP-Training deaktiviert (--disable_mlp).")

    # 6) Metriken speichern
    if all_results:
        metrics_df = pd.DataFrame(all_results)
        metrics_path = os.path.join(args.out_dir, "metrics_all_models.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print_section("Gesamtergebnisse")
        print(metrics_df)
        print(f"\nMetriken gespeichert unter: {metrics_path}")
    else:
        print("Keine Ergebnisse zum Speichern vorhanden.")


if __name__ == "__main__":
    main()