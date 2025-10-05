# scripts/rank_pycaret_models.py
"""
Erstellt ein Ranking aller PyCaret-Modelle über alle Datasets hinweg.
Zeigt das beste Modell pro Dataset und ein Gesamtranking nach mittlerer R2-Performance.
"""

import pandas as pd
import os

def rank_pycaret_models(csv_std, csv_nonstd, out_csv="results/compare/model_ranking_summary.csv"):
    # CSVs laden und kennzeichnen
    df_std = pd.read_csv(csv_std)
    df_std["standardized"] = True

    df_nonstd = pd.read_csv(csv_nonstd)
    df_nonstd["standardized"] = False

    # Beide kombinieren
    df_all = pd.concat([df_std, df_nonstd], ignore_index=True)

    # Nur relevante Spalten behalten
    cols = ["dataset", "Model", "R2", "MAE", "RMSE", "standardized"]
    df_all = df_all[[c for c in cols if c in df_all.columns]]

    # Für jedes Dataset + Standardisierung das beste Modell bestimmen
    best_per_dataset = (
        df_all.sort_values(["dataset", "standardized", "R2"], ascending=[True, True, False])
        .groupby(["dataset", "standardized"])
        .head(1)
        .reset_index(drop=True)
    )

    # Durchschnittliches R² über alle Datasets pro Modell (Gesamtranking)
    avg_r2 = (
        df_all.groupby(["Model", "standardized"])["R2"]
        .mean()
        .reset_index()
        .sort_values(["standardized", "R2"], ascending=[True, False])
    )

    # Ergebnisse speichern
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with pd.ExcelWriter(out_csv.replace(".csv", ".xlsx")) as writer:
        df_all.to_excel(writer, sheet_name="all_results", index=False)
        best_per_dataset.to_excel(writer, sheet_name="best_per_dataset", index=False)
        avg_r2.to_excel(writer, sheet_name="avg_r2_ranking", index=False)

    print("\n=== Beste Modelle pro Dataset ===")
    print(best_per_dataset[["dataset", "Model", "R2", "standardized"]])

    print("\n=== Durchschnittliches R2 nach Modell ===")
    print(avg_r2)

    print(f"\nRanking gespeichert unter: {out_csv.replace('.csv', '.xlsx')}")

if __name__ == "__main__":
    rank_pycaret_models(
        csv_std="results/compare/pycaret_model_comparison_full_std.csv",
        csv_nonstd="results/compare/pycaret_model_comparison_full.csv"
    )