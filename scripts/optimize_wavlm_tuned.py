# scripts/optimize_wavlm_tuned.py

import os
import argparse
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from joblib import parallel_backend
import json


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds, squared=False)
    }


def optimize_model(model_name, X_train, y_train, n_trials=25):
    def objective(trial):
        if model_name == "lightgbm":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 42,
                "n_jobs": -1,
            }
            model = LGBMRegressor(**params)

        elif model_name == "catboost":
            params = {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "iterations": trial.suggest_int("iterations", 200, 800),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.1, 1.0),
                "random_state": 42,
                "verbose": 0,
            }
            model = CatBoostRegressor(**params)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        with parallel_backend("loky"):
            scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=cv, n_jobs=-1)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optimize_wavlm_tuned(dataset_csv, target_col, out_dir, n_trials=25):
    df = pd.read_csv(dataset_csv)
    X = df.drop(columns=[target_col, "filename"], errors="ignore").select_dtypes(include=[np.number])
    y = df[target_col].values

    leak_cols = [c for c in df.columns if c.startswith("wer_") and c != target_col]
    if any(col in X.columns for col in leak_cols):
        raise ValueError(f"Leak detected in {dataset_csv}: {leak_cols}")

    os.makedirs(out_dir, exist_ok=True)
    results = []

    print(f"\n=== Optuna-Tuning: {os.path.basename(dataset_csv)} ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name in ["lightgbm", "catboost"]:
        print(f"\n--- Optimierung: {model_name} ---")
        best_params = optimize_model(model_name, X_train, y_train, n_trials=n_trials)

        print(f"Beste Parameter für {model_name}:")
        print(best_params)

        if model_name == "lightgbm":
            model = LGBMRegressor(**best_params)
        else:
            model = CatBoostRegressor(**best_params)

        with parallel_backend("loky"):
            model.fit(X_train, y_train)

        scores = evaluate_model(model, X_test, y_test)
        results.append({
            "dataset": os.path.basename(dataset_csv),
            "model": model_name,
            "best_r2": scores["r2"],
            "best_mae": scores["mae"],
            "best_rmse": scores["rmse"],
            "params": best_params,
        })

        print(f"Fertig: {model_name} -> R²={scores['r2']:.4f}, MAE={scores['mae']:.4f}, RMSE={scores['rmse']:.4f}")

        with open(os.path.join(out_dir, f"best_params_{model_name}.json"), "w") as f:
            json.dump(best_params, f, indent=2)

    pd.DataFrame(results).to_csv(os.path.join(out_dir, f"optuna_results_{os.path.basename(dataset_csv)}"), index=False)
    print(f"\nErgebnisse gespeichert unter: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=25)
    args = parser.parse_args()

    optimize_wavlm_tuned(args.dataset_csv, args.target_col, args.out_dir, args.n_trials)