# scripts/plot_feature_ranking.py
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main(ranking_csv, out_png):
    # CSV einlesen
    df = pd.read_csv(ranking_csv)

    # Sicherstellen, dass es die richtige Spalte gibt
    if "time_p95_s" not in df.columns or "feature" not in df.columns:
        raise ValueError("CSV muss Spalten 'feature' und 'time_p95_s' enthalten.")

    # Nach Laufzeit sortieren
    df_sorted = df.sort_values("time_p95_s", ascending=True)

    # Plot
    plt.figure(figsize=(10, max(6, len(df_sorted) * 0.3)))
    plt.barh(df_sorted["feature"], df_sorted["time_p95_s"], color="steelblue")
    plt.xlabel("Laufzeit (Sekunden, p95)")
    plt.xscale("log")
    plt.ylabel("Feature")
    plt.title("Benchmark Feature-Laufzeiten (p95 über Dateien)")
    plt.tight_layout()

    # Output speichern
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    print(f"✅ Plot gespeichert unter: {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking_csv", type=str, required=True, help="Pfad zu features_time_ranking.csv")
    parser.add_argument("--out_png", type=str, required=True, help="Pfad zum Ausgabe-PNG")
    args = parser.parse_args()
    main(args.ranking_csv, args.out_png)

#python scripts/plot_feature_ranking.py \
#  --ranking_csv results/benchmark/features_time_ranking.csv \
#  --out_png results/figures/features_time_ranking.png