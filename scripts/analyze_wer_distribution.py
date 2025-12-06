# scripts/analyze_wer_distribution.py
# ------------------------------------------------------
# Analysiert die Verteilung der WER-Werte pro Modell
# (wer_tiny, wer_base, wer_small).
#
# Ausgabe:
#   - Konsolenprint mit Übersicht
#   - CSV mit Verteilungsstatistik:
#       model, threshold_type, threshold, count, share, total, mean, median
#
# Beispielaufruf:
#   python -m scripts.analyze_wer_distribution \
#       --dataset results/datasets/merged_sigmos_wavlm_cv23_balanced_multiwer.csv \
#       --out_csv results/model_comparisons/wer_distribution_cv23_balanced.csv
#
#   python -m scripts.analyze_wer_distribution \
#       --dataset results/datasets/merged_sigmos_wavlm_unseen.csv \
#       --out_csv results/model_comparisons/wer_distribution_unseen.csv
# ------------------------------------------------------

import argparse
import os
import pandas as pd
import numpy as np


def analyze_wer_distribution(df: pd.DataFrame, targets=None):
    if targets is None:
        targets = ["wer_tiny", "wer_base", "wer_small"]

    results = []

    # Schwellen, die wir betrachten wollen
    thresholds = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]

    for target in targets:
        if target not in df.columns:
            print(f"Warnung: Spalte {target} nicht im Datensatz – überspringe.")
            continue

        s = df[target].dropna().astype(float)
        total = len(s)
        if total == 0:
            print(f"Warnung: Spalte {target} enthält nur NaN – überspringe.")
            continue

        mean = float(s.mean())
        median = float(s.median())

        print("\n" + "=" * 72)
        print(f"Verteilung für {target} (n = {total})")
        print(f"  mean   = {mean:.4f}")
        print(f"  median = {median:.4f}")
        print("=" * 72)

        # Für each threshold: Anteil berechnen
        for thr in thresholds:
            if thr == 0.0:
                # Exakt 0
                count = int((s == 0.0).sum())
                threshold_type = "eq"
            else:
                # <= threshold
                count = int((s <= thr).sum())
                threshold_type = "le"

            share = count / total

            results.append(
                {
                    "model": target,
                    "threshold_type": threshold_type,  # "eq" oder "le"
                    "threshold": thr,
                    "count": count,
                    "share": share,
                    "total": total,
                    "mean": mean,
                    "median": median,
                }
            )

            label = "== 0" if thr == 0.0 else f"<= {thr:.2f}"
            print(f"  Anteil WER {label:6s}: {share*100:6.2f}%  (n={count})")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Analysiere WER-Verteilung für wer_tiny/wer_base/wer_small."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Pfad zur CSV mit wer_tiny/wer_base/wer_small.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Pfad zur Ausgabe-CSV mit der Verteilungsstatistik.",
    )
    args = parser.parse_args()

    print(f"Lade Datensatz von: {args.dataset}")
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Shape: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    summary_df = analyze_wer_distribution(df)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    summary_df.to_csv(args.out_csv, index=False)
    print(f"\nVerteilungsstatistik gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()