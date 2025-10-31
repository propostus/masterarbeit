import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler

# ===============================================
# Optuna Objective
# ===============================================

def objective(trial, model_name, X_train, X_val, y_train, y_val):
    """Trainiert ein Modell und gibt R², RMSE, MAE zurück"""
    if model_name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "random_state": 42,
            "verbosity": -1,
        }
        model = LGBMRegressor(**params)
    elif model_name == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": 42,
            "verbose": 0,
        }
        model = CatBoostRegressor(**params)
    else:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    return r2, rmse, mae


def optimize_model(model_name, X_train, X_val, y_train, y_val, trials=8):
    """Optuna-Optimierung"""
    def optuna_objective(trial):
        r2, _, _ = objective(trial, model_name, X_train, X_val, y_train, y_val)
        return -r2

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(optuna_objective, n_trials=trials, show_progress_bar=False)
    best_trial = study.best_trial
    r2, rmse, mae = objective(best_trial, model_name, X_train, X_val, y_train, y_val)
    return r2, rmse, mae, best_trial.params

# ===============================================
# Inkrementelles Experiment
# ===============================================

def run_incremental_feature_blocks(dataset_path, target_col, out_dir, trials=8):
    print(f"=== Starte inkrementelles Feature-Experiment auf {dataset_path} ===")
    df = pd.read_csv(dataset_path)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Spalten, die ausgeschlossen werden
    exclude_cols = ["filename", "snr", "source_embed", "rt60_method", "source_hand"]
    df["base_filename"] = df["filename"].apply(lambda x: x.split("_snr")[0] if "_snr" in x else x)

    # Feature-Gruppen definieren
    feature_groups = {
        "embeddings": [c for c in df.columns if c.isdigit() or c.startswith("mos_")],
        "energy": ["rms_mean", "rms_std", "snr_energy_db", "snr_welch_db"],
        "spectral": [
            "centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std",
            "contrast_band0_mean", "contrast_band1_mean", "flatness_mean", "flatness_std",
            "rolloff_mean", "rolloff_std", "crest_factor", "entropy_mean", "entropy_std"
        ],
        "mfcc": [c for c in df.columns if c.startswith("mfcc")],
        "plp": [c for c in df.columns if c.startswith("plp")],
        "vad_phoneme": ["vad_ratio", "vad_num_active", "vad_total_frames", "phoneme_entropy_mean", "phoneme_entropy_std"],
        "others": ["c50_proxy", "c80_proxy", "tail_duration_s", "lra_db", "srmr_proxy"],
    }

    # GroupSplit vorbereiten
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    base_features = []
    results = []

    for step, (group_name, group_cols) in enumerate(feature_groups.items()):
        base_features += group_cols
        current_features = [c for c in base_features if c in df.columns and c not in exclude_cols + [target_col]]

        X = df[current_features].select_dtypes(include=[float, int, bool])
        y = df[target_col]

        train_idx, val_idx = next(gss.split(X, y, groups=df["base_filename"]))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"\n[Schritt {step}] Füge hinzu: {group_name} ({len(current_features)} Features)")

        for model_name in ["lightgbm", "catboost"]:
            try:
                r2, rmse, mae, params = optimize_model(model_name, X_train, X_val, y_train, y_val, trials)
                print(f"  → {model_name}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
                results.append({
                    "step": step,
                    "group": group_name,
                    "model": model_name,
                    "num_features": len(current_features),
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                })
            except Exception as e:
                print(f"  Fehler bei {model_name} in Schritt {group_name}: {e}")

    # Ergebnisse speichern
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"incremental_feature_blocks_{os.path.basename(dataset_path)}")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nErgebnisse gespeichert unter: {out_path}")


# ===============================================
# CLI Entry Point
# ===============================================

def main():
    parser = argparse.ArgumentParser(description="Inkrementelles Featureblock-Experiment (Embeddings + Handcrafted)")
    parser.add_argument("--dataset", type=str, required=True, help="Pfad zur CSV-Datei (merged SigMOS/WavLM/Features)")
    parser.add_argument("--target", type=str, required=True, help="Zielspalte, z. B. wer_tiny")
    parser.add_argument("--out_dir", type=str, default="results/incremental_feature_blocks")
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()

    run_incremental_feature_blocks(args.dataset, args.target, args.out_dir, args.trials)


if __name__ == "__main__":
    main()