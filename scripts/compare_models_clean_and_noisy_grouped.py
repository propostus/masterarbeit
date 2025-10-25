import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import optuna
from optuna.samplers import TPESampler


def objective(trial, model_name, X_train, X_val, y_train, y_val):
    """Trainiert und bewertet ein Modell basierend auf dem übergebenen Namen."""
    if model_name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
            "random_state": 42,
        }
        model = LGBMRegressor(**params)

    elif model_name == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 5.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": 42,
            "verbose": 0,
        }
        model = CatBoostRegressor(**params)

    elif model_name == "randomforest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "n_jobs": -1,
            "random_state": 42,
        }
        model = RandomForestRegressor(**params)

    elif model_name == "ridge":
        params = {"alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True)}
        model = Ridge(**params, random_state=42)

    elif model_name == "lasso":
        params = {
            "alpha": trial.suggest_float("alpha", 1e-4, 0.5, log=True),
            "max_iter": 5000,
        }
        model = Lasso(**params, random_state=42)

    elif model_name == "mlp":
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (64, 32)]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": 400,
            "random_state": 42,
        }
        model = MLPRegressor(**params)

    else:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    r2 = r2_score(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    return r2, rmse, mae


def optimize_model(model_name, X_train, X_val, y_train, y_val, trials=10):
    """Optuna-Optimierung mit begrenzter Trial-Anzahl."""
    def optuna_objective(trial):
        r2, _, _ = objective(trial, model_name, X_train, X_val, y_train, y_val)
        return -r2  # wir maximieren R²
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(optuna_objective, n_trials=trials, show_progress_bar=False)
    best_trial = study.best_trial
    r2, rmse, mae = objective(best_trial, model_name, X_train, X_val, y_train, y_val)
    return r2, rmse, mae, best_trial.params


def compare_models(dataset_path, target_col, out_dir, trials=10):
    print(f"=== Modellvergleich für {dataset_path} ===")
    df = pd.read_csv(dataset_path)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Features und Zielvariable
    X = df.drop(columns=[target_col, "filename", "snr"], errors="ignore")
    y = df[target_col]

    # Gruppen aus Dateinamen extrahieren (clean/noisy Varianten zusammenhalten)
    groups = df["filename"].str.replace(r"_snr.*", "", regex=True)

    # Gruppiertes Splitten
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print(f"Trainingsdaten: {X_train.shape[0]} | Validierung: {X_val.shape[0]}")

    results = []
    models_no_scale = ["lightgbm", "catboost", "randomforest"]
    models_scale = ["ridge", "lasso", "mlp"]

    for model_name in models_no_scale + models_scale:
        print(f"\n--- {model_name.upper()} ---")
        if model_name in models_scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            r2, rmse, mae, params = optimize_model(model_name, X_train_scaled, X_val_scaled, y_train, y_val, trials)
        else:
            r2, rmse, mae, params = optimize_model(model_name, X_train, X_val, y_train, y_val, trials)

        results.append({"model": model_name, "r2": r2, "rmse": rmse, "mae": mae, "params": params})
        print(f"Ergebnis {model_name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"model_comparison_{os.path.basename(dataset_path)}")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nErgebnisse gespeichert unter: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Vergleiche Modelle auf SigMOS+WavLM clean+noisy Dataset mit gruppiertem Split (clean/noisy getrennt)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/model_comparisons_clean_and_noisy_grouped")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    compare_models(args.dataset, args.target, args.out_dir, args.trials)


if __name__ == "__main__":
    main()