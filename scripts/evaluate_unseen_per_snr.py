# scripts/evaluate_unseen_per_snr.py
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_per_snr(pred_csv, target_col, out_dir):
    print(f"=== Auswertung pro SNR-Stufe für {pred_csv} ===")

    df = pd.read_csv(pred_csv)
    snrs = sorted(df["snr"].unique())
    model_cols = [c for c in df.columns if c.endswith("_pred")]

    print(f"Gefundene Modelle: {', '.join([c.replace('_pred', '') for c in model_cols])}")
    print(f"Gefundene SNR-Stufen: {snrs}")

    results = []
    for snr, group in df.groupby("snr"):
        for col in model_cols:
            model = col.replace("_pred", "")
            y_true = group[target_col]
            y_pred = group[col]

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            results.append({
                "snr": snr,
                "model": model,
                "r2": r2,
                "rmse": rmse,
                "mae": mae
            })

    res_df = pd.DataFrame(results)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"unseen_per_snr_eval_{os.path.basename(pred_csv)}")
    res_df.to_csv(out_path, index=False)

    print(f"\nErgebnisse gespeichert unter: {out_path}")
    print("\n--- R²-Werte pro SNR ---")
    print(res_df.pivot(index="snr", columns="model", values="r2").round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluiert Modelle auf unseen Daten pro SNR-Stufe")
    parser.add_argument("--pred_csv", type=str, required=True, help="Pfad zur CSV mit Predictions")
    parser.add_argument("--target", type=str, required=True, help="Zielspalte (z. B. wer_tiny)")
    parser.add_argument("--out_dir", type=str, default="results/model_comparisons_clean_and_noisy", help="Ausgabeverzeichnis")
    args = parser.parse_args()

    evaluate_per_snr(args.pred_csv, args.target, args.out_dir)