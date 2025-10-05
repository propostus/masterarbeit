# scripts/compare_pycaret_results.py
"""
Vergleicht PyCaret-Ergebnisse zwischen normalisierten und unnormalisierten Setups.
Zeigt Performance-Vergleiche (R2, MAE, RMSE) über alle Datasets hinweg.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_label(csv_path, standardized):
    df = pd.read_csv(csv_path)
    df["standardized"] = standardized
    return df

def compare_pycaret_results(csv_std, csv_nonstd, out_dir="results/figures"):
    os.makedirs(out_dir, exist_ok=True)

    # CSVs laden
    df_std = load_and_label(csv_std, standardized=True)
    df_nonstd = load_and_label(csv_nonstd, standardized=False)

    df_all = pd.concat([df_std, df_nonstd], ignore_index=True)

    # Nur relevante Metriken auswählen
    metrics = ["R2", "MAE", "RMSE"]
    df_filtered = df_all.melt(
        id_vars=["Model", "dataset", "standardized"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )

    # Durchschnittliche Performance nach Modell & Standardisierung
    summary = (
        df_filtered.groupby(["Model", "standardized", "Metric"])["Score"]
        .mean()
        .reset_index()
        .pivot_table(index="Model", columns=["standardized", "Metric"], values="Score")
    )

    print("\n=== Durchschnittliche Modellleistung (Standardisiert vs. Nicht) ===")
    print(summary.round(4))

    # Plot: Effekt der Standardisierung pro Modell (z. B. R2)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_filtered[df_filtered["Metric"] == "R2"],
        x="Model",
        y="Score",
        hue="standardized",
        palette="Blues"
    )
    plt.title("Vergleich der R2-Werte – Standardisiert vs. Nicht")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pycaret_standardization_comparison_r2.png"))
    plt.close()

    print(f"\nAbbildungen gespeichert in {out_dir}")
    print(f"Ergebnisse insgesamt: {len(df_all)} Zeilen")

if __name__ == "__main__":
    compare_pycaret_results(
        csv_std="results/compare/pycaret_model_comparison_full_std.csv",
        csv_nonstd="results/compare/pycaret_model_comparison_full.csv"
    )