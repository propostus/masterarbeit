# scripts/plot_wer_distributions.py
"""
Visualisiert WER-Verteilungen mehrerer Modelle und erzeugt eine Statistik-Tabelle.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_wer_distributions(files, out_path=None):
    plt.figure(figsize=(10, 6))
    all_stats = []

    for file in files:
        model_name = os.path.basename(file).replace("wer_", "").replace("_full.csv", "")
        df = pd.read_csv(file)
        if "wer" not in df.columns:
            raise ValueError(f"Keine Spalte 'wer' in {file}")

        # Plot
        sns.kdeplot(df["wer"], fill=True, alpha=0.3, label=model_name)

        # Statistik berechnen
        stats = df["wer"].describe(percentiles=[0.25, 0.5, 0.75]).to_frame().T
        stats.insert(0, "model", model_name)
        all_stats.append(stats)

    plt.xlim(0, 1.25)
    plt.title("WER-Verteilungen für Common Voice Delta Seg. 20 & 21")
    plt.xlabel("Word Error Rate (WER)")
    plt.ylabel("Dichte")
    plt.legend(title="Whisper Modelle", loc="upper right")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Plot gespeichert unter: {out_path}")

    plt.show()

    # Statistik-Tabelle speichern
    summary = pd.concat(all_stats, ignore_index=True)
    summary_out = os.path.join(os.path.dirname(out_path or files[0]), "wer_summary.csv")
    summary.to_csv(summary_out, index=False)
    print(f"Zusammenfassung gespeichert unter: {summary_out}")
    print(summary[["model", "mean", "std", "min", "25%", "50%", "75%", "max"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="Pfad(e) zu den WER-CSV-Dateien")
    parser.add_argument("--out_path", type=str, default="results/figures/wer_distribution.png")
    args = parser.parse_args()

    plot_wer_distributions(args.files, args.out_path)