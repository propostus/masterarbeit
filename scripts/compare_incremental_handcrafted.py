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

# ============================
# Optuna Objective Definition
# ============================

def objective(trial, model_name, X_train, X_val, y_train, y_val):
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


def optimize_model(model_name, X_train, X_val, y_train, y_val, trials=10):
    def optuna_objective(trial):
        r2, _, _ = objective(trial, model_name, X_train, X_val, y_train, y_val)
        return -r2

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(optuna_objective, n_trials=trials, show_progress_bar=False)
    best_trial = study.best_trial
    r2, rmse, mae = objective(best_trial, model_name, X_train, X_val, y_train, y_val)
    return r2, rmse, mae, best_trial.params

# ============================
# Hauptfunktion: Experiment
# ============================

def run_incremental_experiment(dataset_path, target_col, out_dir, trials=10):
    print(f"=== Starte inkrementelles Experiment auf {dataset_path} ===")
    df = pd.read_csv(dataset_path)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Nicht-numerische / Meta-Spalten ausschließen
    exclude_cols = ["filename", "snr", "source_embed", "rt60_method", "source_hand"]

    # Base-Filename für gruppiertes Splitting
    df["base_filename"] = df["filename"].apply(lambda x: x.split("_snr")[0] if "_snr" in x else x)

    # Alle numerischen Spalten (ohne Zielvariable)
    feature_cols = [c for c in df.columns if c not in exclude_cols + [target_col, "base_filename"]]

    # Split mit Gruppen (kein Data Leakage zwischen SNR-Stufen)
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(df[feature_cols], df[target_col], groups=df["base_filename"]))

    X_train, X_val = df.iloc[train_idx][feature_cols], df.iloc[val_idx][feature_cols]
    y_train, y_val = df.iloc[train_idx][target_col], df.iloc[val_idx][target_col]

    # Optional: sicherstellen, dass X nur numerisch ist
    X_train = X_train.select_dtypes(include=[float, int, bool])
    X_val = X_val.select_dtypes(include=[float, int, bool])

    # Modelle definieren
    models = ["lightgbm", "catboost"]

    results = []
    for model_name in models:
        print(f"\n--- {model_name.upper()} ---")
        r2, rmse, mae, params = optimize_model(model_name, X_train, X_val, y_train, y_val, trials)
        print(f"→ {model_name}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
        results.append({
            "model": model_name,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "params": params
        })

    # Ergebnisse speichern
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"incremental_handcrafted_{os.path.basename(dataset_path)}")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nErgebnisse gespeichert unter: {out_path}")


# ============================
# CLI Entry Point
# ============================

def main():
    parser = argparse.ArgumentParser(description="Inkrementelles Experiment: Embeddings + Handcrafted Features")
    parser.add_argument("--dataset", type=str, required=True, help="Pfad zur CSV-Datei (z. B. merged_sigmos_wavlm_handcrafted_tiny.csv)")
    parser.add_argument("--target", type=str, required=True, help="Zielspalte (z. B. wer_tiny)")
    parser.add_argument("--out_dir", type=str, default="results/incremental_handcrafted")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    run_incremental_experiment(args.dataset, args.target, args.out_dir, args.trials)


if __name__ == "__main__":
    main()