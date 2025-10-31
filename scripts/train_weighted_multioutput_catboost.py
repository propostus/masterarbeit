# scripts/train_weighted_multioutput_catboost.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor


def evaluate_model(y_true, y_pred, label_names):
    results = []
    for i, label in enumerate(label_names):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        results.append({"target": label, "r2": r2, "rmse": rmse, "mae": mae})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Trainiere gewichtetes Multi-Output CatBoost-Modell (WER Tiny/Base/Small)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--weights", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Gewichte für [wer_tiny, wer_base, wer_small]")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== Weighted Multi-Output CatBoost Training auf {args.dataset} ===")
    print(f"Verwendete Gewichte: {args.weights}")

    df = pd.read_csv(args.dataset)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Nicht-numerische Spalten entfernen
    non_numerical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numerical = [c for c in non_numerical if c not in ["filename", "snr"]]
    df = df.drop(columns=non_numerical, errors="ignore")

    target_cols = ["wer_tiny", "wer_base", "wer_small"]
    X = df.drop(columns=target_cols + ["filename", "snr"], errors="ignore")
    y = df[target_cols].values

    # Gruppierter Split nach filename
    groups = df["filename"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Sicherheitscheck
    overlap = set(df.loc[train_idx, "filename"]).intersection(df.loc[val_idx, "filename"])
    if overlap:
        print(f"Warnung: {len(overlap)} Dateien sind in beiden Splits!")
    else:
        print("Split-Check bestanden: Keine Überschneidung zwischen Train- und Validation-Files.")

    # Zielgewichte anwenden: Outputs werden skaliert
    weights = np.array(args.weights)
    y_train_w = y_train * weights
    y_val_w = y_val * weights

    # Modell
    model = CatBoostRegressor(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiRMSE",
        random_seed=42,
        verbose=100
    )

    model.fit(X_train, y_train_w)

    # Rückskalieren der Predictions
    preds = model.predict(X_val) / weights

    results_df = evaluate_model(y_val, preds, target_cols)
    print("\nErgebnisse (Validierung):")
    print(results_df)

    avg_r2 = results_df["r2"].mean()
    avg_rmse = results_df["rmse"].mean()
    avg_mae = results_df["mae"].mean()

    out_path = os.path.join(args.out_dir, "weighted_multioutput_catboost_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nErgebnisse gespeichert unter: {out_path}")

    model_path = os.path.join(args.out_dir, "weighted_multioutput_catboost_model.cbm")
    model.save_model(model_path)
    print(f"Modell gespeichert unter: {model_path}")

    print(f"\nDurchschnittliche Performance: R²={avg_r2:.3f}, RMSE={avg_rmse:.3f}, MAE={avg_mae:.3f}")


if __name__ == "__main__":
    main()