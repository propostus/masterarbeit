# scripts/optimize_sigmos_models.py
import os
import json
import argparse
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lightgbm import LGBMRegressor


def objective(trial, X_train, X_val, y_train, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
    }

    model = LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[],
    )

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse


def optimize_sigmos_models(dataset_path, target_col, trials, out_dir):
    print(f"=== Optimiere Modell für {target_col} auf Basis von {dataset_path} ===")

    df = pd.read_csv(dataset_path)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    if target_col not in df.columns:
        raise ValueError(f"Zielspalte '{target_col}' nicht im Datensatz gefunden.")

    # alle nicht-numerischen Spalten entfernen (z.B. 'filename')
    non_numeric_cols = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric_cols:
        print(f"Ignoriere nicht-numerische Spalten: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=trials)

    best_params = study.best_params
    best_value = study.best_value

    preds = LGBMRegressor(**best_params).fit(X_train, y_train).predict(X_val)
    metrics = {
        "r2": r2_score(y_val, preds),
        "mae": mean_absolute_error(y_val, preds),
        "rmse": mean_squared_error(y_val, preds, squared=False),
    }

    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(study.trials_dataframe()).to_csv(os.path.join(out_dir, "trial_results.csv"), index=False)
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "study_summary.csv"), index=False)

    print(f"Ergebnisse gespeichert unter: {out_dir}")
    print(f"Bestes RMSE: {best_value:.4f}")
    print(f"Bestes R²: {metrics['r2']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Optimiere LightGBM-Modelle auf SigMOS-Embeddings")
    parser.add_argument("--dataset", type=str, required=True, help="Pfad zum Eingabe-Datensatz (CSV)")
    parser.add_argument("--target", type=str, required=True, help="Zielspalte (z.B. wer_tiny)")
    parser.add_argument("--trials", type=int, default=50, help="Anzahl der Trials")
    parser.add_argument("--out_dir", type=str, default="results/sigmos_optuna", help="Ausgabeverzeichnis")

    args = parser.parse_args()
    optimize_sigmos_models(
        dataset_path=args.dataset,
        target_col=args.target,
        trials=args.trials,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()