# scripts/optimize_transformed_target.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.special import boxcox1p

# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def get_model_and_grid(name):
    if name == "ridge":
        return Ridge(), {"model__alpha": [0.1, 1, 10, 100]}
    if name == "lasso":
        return Lasso(max_iter=5000), {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}
    if name == "rf":
        return RandomForestRegressor(random_state=42, n_jobs=-1), {"model__n_estimators": [100, 200]}
    if name == "gbr":
        return GradientBoostingRegressor(random_state=42), {"model__n_estimators": [100, 200]}
    if name == "hgb":
        return HistGradientBoostingRegressor(random_state=42), {"model__max_depth": [None, 5, 10]}
    if name == "svr":
        return SVR(), {"model__C": [0.1, 1, 10]}
    raise ValueError(f"Unbekanntes Modell: {name}")

def get_transformer(name):
    if name == "none":
        return None
    if name == "yeojohnson":
        return PowerTransformer(method="yeo-johnson")
    if name == "log1p":
        return FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)
    if name == "boxcox":
        return FunctionTransformer(lambda x: boxcox1p(x - np.min(x) + 1e-6, 0.15))
    raise ValueError(f"Unbekannte Transformation: {name}")

# ---------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------

def optimize_models(dataset_csv, target_col, models, transforms, out_dir):
    print(f"\nLade Dataset: {dataset_csv}")
    df = pd.read_csv(dataset_csv)

    # Nur numerische Features behalten, filename und Ziel entfernen
    X = df.drop(columns=[target_col, "filename"], errors="ignore").select_dtypes(include=[np.number])
    y = df[target_col].values

    # Leere oder NaN-Zielwerte entfernen
    valid = ~np.isnan(y)
    X, y = X.loc[valid], y[valid]

    # Train/Test-Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"summary_{os.path.basename(dataset_csv)}")

    results = []

    for transform in transforms:
        print(f"\n=== Transformation: {transform} ===")
        for model_name in models:
            print(f"--- Modell: {model_name} ---")
            try:
                model, param_grid = get_model_and_grid(model_name)
                transformer = get_transformer(transform)

                # Pipeline + optional Transformierte Zielvariable
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", model)
                ])

                if transformer is not None:
                    reg = TransformedTargetRegressor(regressor=pipe, transformer=transformer)
                    # Anpassen der Parameternamen für GridSearchCV
                    param_grid = {f"regressor__{k}": v for k, v in param_grid.items()}
                else:
                    reg = pipe

                # Cross-Validation auf Trainingsdaten
                grid = GridSearchCV(
                    reg,
                    param_grid=param_grid,
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                    scoring="r2",
                    n_jobs=-1
                )
                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_
                preds = best_model.predict(X_test)

                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                rmse = mean_squared_error(y_test, preds, squared=False)

                results.append({
                    "dataset": os.path.basename(dataset_csv),
                    "model": model_name,
                    "transform": transform,
                    "best_r2": r2,
                    "best_mae": mae,
                    "best_rmse": rmse,
                    "params": grid.best_params_
                })

                print(f"Fertig: {model_name} ({transform}) -> R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

            except Exception as e:
                print(f"Fehler bei Modell '{model_name}' mit Transform '{transform}': {e}")
                results.append({
                    "dataset": os.path.basename(dataset_csv),
                    "model": model_name,
                    "transform": transform,
                    "best_r2": np.nan,
                    "best_mae": np.nan,
                    "best_rmse": np.nan,
                    "params": {}
                })

    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\nSummary gespeichert unter: {summary_path}")

# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--target_col", type=str, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--transforms", nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    optimize_models(args.dataset_csv, args.target_col, args.models, args.transforms, args.out_dir)