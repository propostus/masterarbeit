# scripts/compare_models_tiny_full.py
"""
Vergleicht verschiedene Regressionsmodelle auf allen Feature-Selection-Datasets
(z. B. für WER tiny full). Beinhaltet automatische NaN-Behandlung und nur numerische Features.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor

def evaluate_model(model, X, y, cv_splits=5):
    """Führt Cross-Validation durch und gibt Metriken zurück."""
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
    mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error"))

    return {
        "r2_mean": np.mean(r2),
        "r2_std": np.std(r2),
        "mae_mean": np.mean(mae),
        "mae_std": np.std(mae),
        "rmse_mean": np.mean(rmse),
        "rmse_std": np.std(rmse),
    }

def compare_models_on_dataset(csv_path, target_col):
    """Lädt ein Dataset, bereitet es vor und vergleicht Modelle."""
    df = pd.read_csv(csv_path)
    print(f"=== Vergleiche Modelle auf {os.path.basename(csv_path)} ===")

    # Zielspalte prüfen
    if target_col not in df.columns:
        raise ValueError(f"Spalte '{target_col}' nicht im Dataset {csv_path}")

    # Numerische Features auswählen
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
    y = df[target_col].astype(float)

    # NaN-Werte imputieren
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    models = {
        "rf": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "hgb": HistGradientBoostingRegressor(random_state=42),
        "lgbm": LGBMRegressor(random_state=42),
        "mlp": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "svr": SVR(),
        "ridge": Ridge(alpha=1.0),
    }

    results = []
    for name, model in models.items():
        print(f"→ Trainiere {name}")
        try:
            metrics = evaluate_model(model, X, y)
            metrics.update({"model": name})
            results.append(metrics)
        except Exception as e:
            print(f"Fehler bei {name} auf {os.path.basename(csv_path)}: {e}")
            continue

    df_res = pd.DataFrame(results)
    df_res["dataset"] = os.path.basename(csv_path)
    return df_res

def run_comparison(dataset_dir, out_csv, target_col="wer"):
    """Vergleicht Modelle über alle Datasets in einem Ordner."""
    csv_files = sorted([os.path.join(dataset_dir, f)
                        for f in os.listdir(dataset_dir)
                        if f.endswith(".csv")])

    all_results = []
    for csv_path in csv_files:
        try:
            df_res = compare_models_on_dataset(csv_path, target_col)
            all_results.append(df_res)
        except Exception as e:
            print(f"Fehler bei {csv_path}: {e}")

    if not all_results:
        print("Keine Ergebnisse generiert.")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_all.to_csv(out_csv, index=False)
    print(f"\nGesamtergebnis gespeichert unter: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--target_col", default="wer")
    args = parser.parse_args()

    run_comparison(args.dataset_dir, args.out_csv, args.target_col)