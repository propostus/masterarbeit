# scripts/optimize_sigmos_wavlm_models.py
import os
import json
import pandas as pd
import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def optimize_sigmos_wavlm_models(dataset, target, trials=100, out_dir="results/sigmos_wavlm_optuna"):
    print(f"=== Optimiere Modell für {target} auf Basis von {dataset} ===")

    # Daten laden
    df = pd.read_csv(dataset)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Ziel und Features
    y = df[target]
    X = df.drop(columns=["filename", target], errors="ignore")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Objective-Funktion
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse", callbacks=[])

        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        return rmse

    # Optuna-Studie
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)

    # Bestes Modell bewerten
    best_params = study.best_params
    model = LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse = mean_squared_error(y_val, preds, squared=False)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    # Ergebnisse speichern
    os.makedirs(out_dir, exist_ok=True)
    results = pd.DataFrame([{"r2": r2, "rmse": rmse, "mae": mae}])
    results.to_csv(os.path.join(out_dir, "study_summary.csv"), index=False)

    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Ergebnisse gespeichert unter: {out_dir}")
    print(f"Bestes RMSE: {rmse:.4f}")
    print(f"Bestes R²:  {r2:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optimiere SigMOS+WavLM Modell")
    parser.add_argument("--dataset", type=str, required=True, help="Pfad zur CSV-Datei")
    parser.add_argument("--target", type=str, required=True, help="Zielspalte (z. B. wer_tiny)")
    parser.add_argument("--trials", type=int, default=100, help="Anzahl Trials")
    parser.add_argument("--out_dir", type=str, required=True, help="Output-Verzeichnis")

    args = parser.parse_args()
    optimize_sigmos_wavlm_models(
        dataset=args.dataset,
        target=args.target,
        trials=args.trials,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()