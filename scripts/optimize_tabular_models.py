# scripts/optimize_tabular_models.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from joblib import parallel_backend


def get_model_and_grid(name):
    """Gibt Modell und zugehöriges GridSearch-Parameter-Set zurück."""
    if name == "lightgbm":
        return LGBMRegressor(random_state=42, n_jobs=-1), {
            "model__num_leaves": [31, 63],
            "model__learning_rate": [0.05, 0.1],
            "model__n_estimators": [100, 300],
        }

    elif name == "catboost":
        return CatBoostRegressor(random_state=42, verbose=0), {
            "model__depth": [6, 8],
            "model__learning_rate": [0.05, 0.1],
            "model__iterations": [200, 400],
        }

    else:
        raise ValueError(f"Unbekanntes Modell: {name}")


def evaluate_model(model, X_test, y_test):
    """Berechnet R², MAE und RMSE auf Testdaten."""
    preds = model.predict(X_test)
    return {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds, squared=False)
    }


def optimize_tabular_models(dataset_csv, target_col, models, out_dir):
    """Trainiert LightGBM und CatBoost mit GridSearchCV auf gegebene Datensätze."""
    df = pd.read_csv(dataset_csv)
    X = df.drop(columns=[target_col, "filename"], errors="ignore").select_dtypes(include=[np.number])
    y = df[target_col].values

    os.makedirs(out_dir, exist_ok=True)
    results = []

    print(f"\n=== Dataset: {os.path.basename(dataset_csv)} ===")

    # Train/Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name in models:
        print(f"--- Modell: {model_name} ---")
        model, param_grid = get_model_and_grid(model_name)

        # StandardScaler nur bei Nicht-Tree-Modellen
        steps = []
        if model_name not in ["lightgbm", "catboost"]:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))
        pipe = Pipeline(steps)

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring="r2",
            n_jobs=-1,
        )

        with parallel_backend("loky"):
            grid.fit(X_train, y_train)

        scores = evaluate_model(grid.best_estimator_, X_test, y_test)
        results.append({
            "dataset": os.path.basename(dataset_csv),
            "model": model_name,
            "best_r2": scores["r2"],
            "best_mae": scores["mae"],
            "best_rmse": scores["rmse"],
            "params": grid.best_params_,
        })

        print(f"Fertig: {model_name} -> R²={scores['r2']:.4f}, MAE={scores['mae']:.4f}, RMSE={scores['rmse']:.4f}")

    out_path = os.path.join(out_dir, f"tabular_{os.path.basename(dataset_csv)}")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Ergebnisse gespeichert unter: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=["lightgbm", "catboost"])
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    optimize_tabular_models(args.dataset_csv, args.target_col, args.models, args.out_dir)