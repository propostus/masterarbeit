# scripts/evaluate_unseen_per_snr_and_model.py
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_per_snr_and_model(pred_csv, target_col, out_dir):
    print(f"=== Auswertung pro SNR-Stufe und Modell für {pred_csv} ===")

    df = pd.read_csv(pred_csv)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen")

    # Prüfe, ob die Zielspalte existiert
    if target_col not in df.columns:
        print(f"Warnung: Spalte '{target_col}' nicht gefunden. Versuche stattdessen 'y_true'.")
        target_col = "y_true"

    if "y_pred" not in df.columns or "snr" not in df.columns or "model" not in df.columns:
        raise ValueError("CSV muss die Spalten ['snr', 'model', 'y_pred'] enthalten.")

    # Gruppiere nach SNR und Modell
    grouped = df.groupby(["snr", "model"])
    results = []

    for (snr, model), group in grouped:
        y_true = group[target_col]
        y_pred = group["y_pred"]

        # Berechne Kennzahlen
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        results.append({
            "snr": snr,
            "model": model,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "n_samples": len(group)
        })

        print(f"SNR={snr} | Modell={model} | R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f} ({len(group)} Samples)")

    # Ergebnisse speichern
    results_df = pd.DataFrame(results)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "unseen_eval_per_snr_and_model.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nErgebnisse gespeichert unter: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluiert Modelle pro SNR-Stufe auf unseen data.")
    parser.add_argument("--pred_csv", type=str, required=True, help="Pfad zur Predictions-CSV")
    parser.add_argument("--target", type=str, default="y_true", help="Name der Zielspalte")
    parser.add_argument("--out_dir", type=str, required=True, help="Ordner für Ausgabedatei")
    args = parser.parse_args()

    evaluate_per_snr_and_model(args.pred_csv, args.target, args.out_dir)


if __name__ == "__main__":
    main()