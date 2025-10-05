# scripts/train_baseline.py
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
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
    """Wähle Modell basierend auf dem Typ."""
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "hgb":
        return HistGradientBoostingRegressor(
            max_depth=None,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42
        )
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
            raise ImportError("LightGBM ist nicht installiert. Installiere es mit `pip install lightgbm`.")
        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unbekanntes Modell: {model_type}")


def train_baseline(dataset_csv, out_dir, n_splits=5, model_type="rf"):
    # Daten laden
    df = pd.read_csv(dataset_csv)

    if "wer" not in df.columns:
        raise ValueError("Die CSV muss eine Spalte 'wer' enthalten.")

    # Features und Ziel trennen
    df_y = df["wer"]
    df_X = df.drop(columns=["filename", "wer"], errors="ignore")

    # Nur numerische Spalten verwenden
    X = df_X.select_dtypes(include=[np.number])
    y = df_y.values

    print(f"Gesamtspalten: {df_X.shape[1]}, genutzt (nur numerisch): {X.shape[1]}")
    print(f"Anzahl NaNs im Feature-Set: {X.isna().sum().sum()}")

    # Modell wählen
    model = get_model(model_type)

    # Pipeline
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", model)
    ])

    # Cross-Validation
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_train_score=False
    )

    # Ergebnisse
    results = {
        "model": model_type,
        "r2_mean": np.mean(scores["test_r2"]),
        "r2_std": np.std(scores["test_r2"]),
        "mae_mean": -np.mean(scores["test_neg_mean_absolute_error"]),
        "mae_std": np.std(-scores["test_neg_mean_absolute_error"]),
        "rmse_mean": -np.mean(scores["test_neg_root_mean_squared_error"]),
        "rmse_std": np.std(-scores["test_neg_root_mean_squared_error"]),
    }

    print("Cross-Validation Ergebnisse:")
    for k, v in results.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Speichern
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([results]).to_csv(
        os.path.join(out_dir, f"{model_type}_metrics.csv"), index=False
    )

    print(f"Ergebnisse gespeichert in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True, help="Pfad zu training_dataset.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Ordner für Ergebnisse")
    parser.add_argument("--n_splits", type=int, default=5, help="Anzahl CV-Splits (default=5)")
    parser.add_argument("--model", type=str, default="rf",
                        choices=["rf", "hgb", "mlp", "ridge", "lasso", "elasticnet", "svr", "dummy", "lgbm"],
                        help="Modelltyp")
    args = parser.parse_args()

    train_baseline(args.dataset_csv, args.out_dir, n_splits=args.n_splits, model_type=args.model)