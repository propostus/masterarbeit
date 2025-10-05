# scripts/compare_models.py
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor

# Optional: LightGBM
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def get_model(model_type: str):
    """Wähle Modell basierend auf Typ"""
    if model_type == "rf":
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    elif model_type == "hgb":
        return HistGradientBoostingRegressor(learning_rate=0.1, random_state=42)
    elif model_type == "mlp":
        return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    elif model_type == "ridge":
        return Ridge(alpha=1.0, random_state=42)
    elif model_type == "lasso":
        return Lasso(alpha=0.001, max_iter=5000, random_state=42)
    elif model_type == "elasticnet":
        return ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=42)
    elif model_type == "svr":
        return SVR(kernel="rbf", C=1.0, epsilon=0.1)
    elif model_type == "dummy":
        return DummyRegressor(strategy="mean")
    elif model_type == "lgbm":
        if not HAS_LGBM:
            raise ImportError("LightGBM ist nicht installiert. Installiere mit `pip install lightgbm`.")
        return LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=-1, random_state=42, n_jobs=-1
        )
    else:
        raise ValueError(f"Unbekanntes Modell: {model_type}")


def train_and_evaluate(X, y, model_type="rf", n_splits=5):
    model = get_model(model_type)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_train_score=False
    )

    return {
        "model": model_type,
        "r2_mean": np.mean(scores["test_r2"]),
        "r2_std": np.std(scores["test_r2"]),
        "mae_mean": -np.mean(scores["test_neg_mean_absolute_error"]),
        "mae_std": np.std(-scores["test_neg_mean_absolute_error"]),
        "rmse_mean": -np.mean(scores["test_neg_root_mean_squared_error"]),
        "rmse_std": np.std(-scores["test_neg_root_mean_squared_error"]),
    }


def compare_datasets(dataset_paths, out_csv, n_splits=5):
    results = []

    # Liste aller Modelle
    models = ["rf", "hgb", "mlp", "ridge", "lasso", "elasticnet", "svr", "dummy"]
    if HAS_LGBM:
        models.append("lgbm")

    for dataset_path in dataset_paths:
        df = pd.read_csv(dataset_path)
        if "wer" not in df.columns:
            raise ValueError(f"Dataset {dataset_path} hat keine 'wer'-Spalte")

        X = df.drop(columns=["filename", "wer"], errors="ignore").select_dtypes(include=[np.number])
        y = df["wer"].values

        dataset_name = os.path.basename(dataset_path).replace(".csv", "")
        print(f"\n==> Dataset: {dataset_name} ({X.shape[0]} Samples, {X.shape[1]} Features)")

        for model_type in models:
            try:
                res = train_and_evaluate(X, y, model_type=model_type, n_splits=n_splits)
                res["dataset"] = dataset_name
                results.append(res)
                print(f"  {model_type:10s} | R2={res['r2_mean']:.4f}, MAE={res['mae_mean']:.4f}, RMSE={res['rmse_mean']:.4f}")
            except Exception as e:
                print(f"  {model_type:10s} | Fehler: {e}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nAlle Ergebnisse gespeichert in: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Liste von Datasets (CSV-Dateien)")
    parser.add_argument("--out_csv", type=str, required=True, help="Pfad für Ergebnis-Tabelle")
    parser.add_argument("--n_splits", type=int, default=5, help="Cross-Validation Splits (Default: 5)")
    args = parser.parse_args()

    compare_datasets(args.datasets, args.out_csv, n_splits=args.n_splits)