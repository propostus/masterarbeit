# scripts/optimize_models.py
"""
Optimiert ausgewählte Modelle (z. B. Ridge, HGB) mit GridSearchCV
auf einem gegebenen Feature-Dataset.
Speichert beste Parameter, Scores und Ergebnisse.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error


def optimize_model(dataset_csv, target_col, model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Optimierung für {model_name} auf {os.path.basename(dataset_csv)} ===")

    df = pd.read_csv(dataset_csv)
    df = df.select_dtypes(include=[np.number])  # nur numerische Spalten
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Cross-Validation Setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Scoring
    scoring = {
        "r2": make_scorer(r2_score),
        "mae": make_scorer(mean_absolute_error),
        "rmse": make_scorer(mean_squared_error, squared=False),
    }

    # Modell & Parameter-Grid
    if model_name == "ridge":
        model = Ridge()
        param_grid = {
            "model__alpha": [0.001, 0.01, 0.1, 1, 5, 10, 50, 100],
        }

    elif model_name == "hgb":
        model = HistGradientBoostingRegressor(random_state=42)
        param_grid = {
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7, None],
            "model__max_iter": [100, 200, 300],
        }

    else:
        raise ValueError(f"Unbekanntes Modell: {model_name}")

    # Pipeline
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # Grid Search
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring=scoring,
        refit="r2",
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    grid.fit(X, y)

    # Ergebnisse
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_r2 = grid.best_score_

    print(f"\nBeste Parameter: {best_params}")
    print(f"Bester R²: {best_r2:.4f}")

    # Crossval-Ergebnisse speichern
    results_df = pd.DataFrame(grid.cv_results_)
    out_path = os.path.join(
        out_dir,
        f"opt_{model_name}_{os.path.basename(dataset_csv).replace('.csv', '')}.csv"
    )
    results_df.to_csv(out_path, index=False)

    print(f"Gesamte Ergebnisse gespeichert unter: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--model", required=True, choices=["ridge", "hgb"])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    optimize_model(args.dataset_csv, args.target_col, args.model, args.out_dir)