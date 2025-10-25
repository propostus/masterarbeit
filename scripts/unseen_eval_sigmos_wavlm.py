import os
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import optuna
from optuna.samplers import TPESampler


def get_model_and_space(model_name, trial):
    if model_name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state": 42,
        }
        return LGBMRegressor(**params), False

    if model_name == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 300, 800),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": 42,
            "verbose": 0,
        }
        return CatBoostRegressor(**params), False

    if model_name == "randomforest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42,
            "n_jobs": -1,
        }
        return RandomForestRegressor(**params), False

    if model_name == "ridge":
        params = {"alpha": trial.suggest_float("alpha", 1e-3, 5.0, log=True)}
        return Ridge(**params, random_state=42), True

    if model_name == "lasso":
        params = {"alpha": trial.suggest_float("alpha", 1e-3, 0.5, log=True), "max_iter": 5000}
        return Lasso(**params, random_state=42), True

    if model_name == "mlp":
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (64, 32)]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True),
            "max_iter": 400,
            "random_state": 42,
        }
        return MLPRegressor(**params), True

    raise ValueError(f"Unbekanntes Modell: {model_name}")


def evaluate(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
    }


def run_unseen_eval(dataset_csv, target_col, out_dir, trials=20, test_size=0.2, random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(dataset_csv)
    # Falls vorhanden: Leaks entfernen
    drop_cols = [c for c in ["filename", "snr", "reference", "hypothesis"] if c in df.columns]
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]
    groups = df["filename"] if "filename" in df.columns else pd.Series(np.arange(len(df)))

    # Gruppierter Hold-out-Split: gesamte Datei-Familie bleibt zusammen
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx].values, X.iloc[test_idx].values
    y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
    meta_test = df.iloc[test_idx][["filename"] + ([ "snr"] if "snr" in df.columns else [])].copy()

    models_no_scale = ["lightgbm", "catboost", "randomforest"]
    models_scale = ["ridge", "lasso", "mlp"]
    all_models = models_no_scale + models_scale

    results = []
    preds_frames = []

    # Inneres Val-Split innerhalb des Trainings für Optuna
    Xtr, Xval, ytr, yval = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    for model_name in all_models:
        def optuna_objective(trial):
            model, needs_scale = get_model_and_space(model_name, trial)
            if needs_scale:
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr)
                Xval_s = scaler.transform(Xval)
                model.fit(Xtr_s, ytr)
                preds = model.predict(Xval_s)
            else:
                model.fit(Xtr, ytr)
                preds = model.predict(Xval)
            return -r2_score(yval, preds)

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=random_state))
        study.optimize(optuna_objective, n_trials=trials, show_progress_bar=False)

        best_params = study.best_trial.params
        # Reinstanziieren mit Bestparams
        trial = optuna.trial.FixedTrial(best_params)
        best_model, needs_scale = get_model_and_space(model_name, trial)

        if needs_scale:
            scaler = StandardScaler()
            Xtr_full = scaler.fit_transform(X_train)
            Xte_full = scaler.transform(X_test)
            best_model.fit(Xtr_full, y_train)
            y_pred = best_model.predict(Xte_full)
        else:
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

        mets = evaluate(y_test, y_pred)

        results.append({
            "dataset": os.path.basename(dataset_csv),
            "target": target_col,
            "model": model_name,
            "trials": trials,
            "test_size": test_size,
            "r2": mets["r2"],
            "rmse": mets["rmse"],
            "mae": mets["mae"],
            "best_params": json.dumps(best_params),
        })

        pf = meta_test.copy()
        pf["y_true"] = y_test
        pf["y_pred"] = y_pred
        pf["model"] = model_name
        preds_frames.append(pf)

        print(f"{model_name}: R²={mets['r2']:.4f}  RMSE={mets['rmse']:.4f}  MAE={mets['mae']:.4f}")

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(out_dir, f"unseen_eval_{os.path.basename(dataset_csv)}")
    res_df.to_csv(out_csv, index=False)

    preds_df = pd.concat(preds_frames, axis=0, ignore_index=True)
    preds_out = os.path.join(out_dir, f"unseen_predictions_{os.path.basename(dataset_csv)}")
    preds_df.to_csv(preds_out, index=False)

    print(f"\nMetriken gespeichert: {out_csv}")
    print(f"Predictions gespeichert: {preds_out}")


def main():
    parser = argparse.ArgumentParser(description="Grouped unseen evaluation für SigMOS+WavLM Datasets (clean+noisy).")
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    run_unseen_eval(
        dataset_csv=args.dataset_csv,
        target_col=args.target_col,
        out_dir=args.out_dir,
        trials=args.trials,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()