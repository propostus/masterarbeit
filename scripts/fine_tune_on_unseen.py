# scripts/fine_tune_on_unseen.py
# ------------------------------------------------------
# Fine-Tuning bestehender LightGBM- und CatBoost-Modelle
# auf kleinem Teil der Unseen-Daten (z. B. 10 %)
# mit robustem Feature-Alignment und Early Stopping.
# ------------------------------------------------------

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor

# -----------------------
# CCC-Metrik
# -----------------------
def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return (2 * cov) / denom if denom > 0 else 0.0


# -----------------------
# Feature-Alignment
# -----------------------
def align_features(df_X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    if not feature_names:
        return df_X.copy()  # falls das Modell keine Featureliste hat
    common = [f for f in feature_names if f in df_X.columns]
    if len(common) == 0:
        print("⚠️  Warnung: Keine übereinstimmenden Feature-Namen gefunden – verwende alle numerischen Spalten.")
        return df_X.copy()
    X = df_X.reindex(columns=feature_names, fill_value=0.0)
    return X.astype(np.float32)


# -----------------------
# LightGBM Fine-Tuning
# -----------------------
def fine_tune_and_eval_lgbm(trained_model: LGBMRegressor,
                            X_ft: pd.DataFrame, y_ft: np.ndarray,
                            X_ev: pd.DataFrame, y_ev: np.ndarray,
                            out_path: str):
    params = trained_model.get_params(deep=True)
    params.update(dict(
        n_estimators=300,
        learning_rate=min(params.get("learning_rate", 0.05) * 0.5, 0.05),
        random_state=42
    ))

    ft_model = LGBMRegressor(**params)
    ft_model.fit(
        X_ft, y_ft,
        init_model=trained_model,
        eval_set=[(X_ev, y_ev)],
        callbacks=[early_stopping(stopping_rounds=30, verbose=False), log_evaluation(period=0)]
    )

    preds = ft_model.predict(X_ev)
    return (
        r2_score(y_ev, preds),
        mean_squared_error(y_ev, preds, squared=False),
        mean_absolute_error(y_ev, preds),
        concordance_correlation_coefficient(y_ev, preds)
    )


# -----------------------
# CatBoost Fine-Tuning
# -----------------------
def fine_tune_and_eval_cat(cat_model: CatBoostRegressor,
                           X_ft: pd.DataFrame, y_ft: np.ndarray,
                           X_ev: pd.DataFrame, y_ev: np.ndarray,
                           out_path: str):
    params = cat_model.get_params()
    # Entferne random_state, sonst Fehler
    params.pop("random_state", None)

    params.update(dict(
        iterations=300,
        learning_rate=min(params.get("learning_rate", 0.03) * 0.7, 0.03),
        depth=params.get("depth", 8),
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    ))

    ft_model = CatBoostRegressor(**params)
    ft_model.fit(
        X_ft, y_ft,
        init_model=cat_model,
        eval_set=(X_ev, y_ev),
        use_best_model=True,
        verbose=False
    )

    preds = ft_model.predict(X_ev)
    return (
        r2_score(y_ev, preds),
        mean_squared_error(y_ev, preds, squared=False),
        mean_absolute_error(y_ev, preds),
        concordance_correlation_coefficient(y_ev, preds)
    )


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune trained models on small fraction of unseen data (robust feature alignment)")
    parser.add_argument("--unseen_csv", required=True, help="Pfad zum Unseen-Datensatz (CSV)")
    parser.add_argument("--models_dir", required=True, help="Verzeichnis mit trainierten Modellen")
    parser.add_argument("--out_dir", required=True, help="Zielverzeichnis für Fine-Tuned-Modelle & Ergebnisse")
    parser.add_argument("--fraction", type=float, default=0.1, help="Anteil der Unseen-Daten zum Fine-Tunen (0-1)")
    parser.add_argument("--random_state", type=int, default=42, help="Seed für Reproduzierbarkeit")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.unseen_csv, low_memory=False)
    print(f"=== Unseen-Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten ===")

    targets = ["wer_tiny", "wer_base", "wer_small"]
    exclude_cols = ["filename", "reference", "hypothesis", "source", "group_id"] + targets
    candidate_features = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    df_all = df.dropna(subset=targets).copy()
    X_all = df_all[candidate_features].astype(np.float32)
    y_all = df_all[targets].copy()

    X_ft, X_ev, y_ft, y_ev = train_test_split(
        X_all, y_all, test_size=(1 - args.fraction), random_state=args.random_state
    )

    results = []

    for target in targets:
        print(f"\n=== Fine-Tuning für Zielvariable: {target} ===")

        # LightGBM
        lgb_path = os.path.join(args.models_dir, f"LightGBM_{target}.pkl")
        if os.path.exists(lgb_path):
            print(f"→ Lade LightGBM: {lgb_path}")
            trained = joblib.load(lgb_path)

            train_features = getattr(trained, "feature_name_", [])
            X_ft_lgb = align_features(X_ft, train_features)
            X_ev_lgb = align_features(X_ev, train_features)

            out_path = os.path.join(args.out_dir, f"LightGBM_{target}_finetuned.pkl")
            r2, rmse, mae, ccc = fine_tune_and_eval_lgbm(
                trained, X_ft_lgb, y_ft[target].values, X_ev_lgb, y_ev[target].values, out_path
            )
            results.append({"model": "LightGBM", "target": target, "r2": r2, "rmse": rmse, "mae": mae, "ccc": ccc})

        # CatBoost
        cat_path = os.path.join(args.models_dir, f"CatBoost_{target}.pkl")
        if os.path.exists(cat_path):
            print(f"→ Lade CatBoost: {cat_path}")
            trained = joblib.load(cat_path)

            train_features = getattr(trained, "feature_names_", [])
            X_ft_cat = align_features(X_ft, train_features)
            X_ev_cat = align_features(X_ev, train_features)

            out_path = os.path.join(args.out_dir, f"CatBoost_{target}_finetuned.pkl")
            r2, rmse, mae, ccc = fine_tune_and_eval_cat(
                trained, X_ft_cat, y_ft[target].values, X_ev_cat, y_ev[target].values, out_path
            )
            results.append({"model": "CatBoost", "target": target, "r2": r2, "rmse": rmse, "mae": mae, "ccc": ccc})

    res_df = pd.DataFrame(results)
    out_csv = os.path.join(args.out_dir, "fine_tune_results.csv")
    res_df.to_csv(out_csv, index=False)

    print("\n=== Fine-Tuning abgeschlossen ===")
    print(res_df)
    print(f"\nErgebnisse gespeichert unter: {out_csv}")


if __name__ == "__main__":
    main()