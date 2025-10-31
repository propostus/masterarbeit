# scripts/train_multioutput_catboost.py

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor


def evaluate_model(y_true, y_pred, label_names):
    """Berechnet R², RMSE, MAE pro Zielvariable."""
    results = []
    for i, label in enumerate(label_names):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        results.append({"target": label, "r2": r2, "rmse": rmse, "mae": mae})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Trainiere Multi-Output CatBoost-Modell für WER-Tiny/Base/Small")
    parser.add_argument("--dataset", type=str, required=True, help="Pfad zur gemergten CSV mit allen WER-Zielen")
    parser.add_argument("--out_dir", type=str, required=True, help="Pfad zum Speichern der Ergebnisse")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"=== Multi-Output CatBoost Training auf {args.dataset} ===")

    # Daten laden
    df = pd.read_csv(args.dataset)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # Nicht-numerische Spalten entfernen (außer filename, snr)
    non_numerical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numerical = [c for c in non_numerical if c not in ["filename", "snr"]]
    df = df.drop(columns=non_numerical, errors="ignore")

    # Zielspalten definieren
    target_cols = ["wer_tiny", "wer_base", "wer_small"]
    X = df.drop(columns=target_cols + ["filename", "snr"], errors="ignore")
    y = df[target_cols].values

    # Gruppierung nach filename (damit clean + noisy Varianten zusammenbleiben)
    groups = df["filename"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"Trainingsset: {X_train.shape}, Validierungsset: {X_val.shape}")

    # === Sicherheitsprüfung: kein File darf in beiden Splits vorkommen ===
    train_files = set(df.loc[train_idx, "filename"])
    val_files = set(df.loc[val_idx, "filename"])
    overlap = train_files.intersection(val_files)
    if len(overlap) > 0:
        print(f"Warnung: {len(overlap)} Dateien sind in beiden Splits! Das deutet auf Leakage hin.")
        print(sorted(list(overlap))[:10])  # nur erste 10 anzeigen
    else:
        print("Split-Check bestanden: Keine Überschneidung zwischen Train- und Validation-Files.")

    # Modell initialisieren (Multi-Output)
    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiRMSE",
        random_seed=42,
        verbose=100
    )

    model.fit(X_train, y_train)

    # Vorhersagen & Bewertung
    preds = model.predict(X_val)
    results_df = evaluate_model(y_val, preds, target_cols)
    print("\nErgebnisse auf Validierungsdaten:")
    print(results_df)

    # Durchschnittliche Performance
    avg_r2 = results_df["r2"].mean()
    avg_rmse = results_df["rmse"].mean()
    avg_mae = results_df["mae"].mean()

    # Ergebnisse speichern
    out_path = os.path.join(args.out_dir, "multioutput_catboost_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nErgebnisse gespeichert unter: {out_path}")

    # Modell speichern
    model_out_path = os.path.join(args.out_dir, "multioutput_catboost_model.cbm")
    model.save_model(model_out_path)
    print(f"Modell gespeichert unter: {model_out_path}")

    print(f"\nDurchschnittliche Performance: R²={avg_r2:.3f}, RMSE={avg_rmse:.3f}, MAE={avg_mae:.3f}")


if __name__ == "__main__":
    main()