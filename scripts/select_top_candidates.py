# scripts/select_top_candidates.py
import argparse
import pandas as pd
import os

def select_top_candidates(summary_csv, out_csv, top_n=10):
    df = pd.read_csv(summary_csv)

    # Schema-Erkennung
    if {"best_r2", "best_mae", "best_rmse"}.issubset(df.columns):
        # Schema aus summarize_optimizations.py
        sort_col = "best_r2"
        sel_cols = ["dataset", "model", "best_r2", "best_mae", "best_rmse"]
        pretty = df[sel_cols].copy()
    elif {"r2_mean", "mae_mean", "rmse_mean"}.issubset(df.columns):
        # Schema aus compare_models_*.py
        # Vereinheitlichen der Spaltennamen für einheitlichen Output
        pretty = df.copy()
        # falls es die Spalte 'features' gibt, hänge sie an den Datasetnamen an
        if "features" in pretty.columns and "dataset" in pretty.columns:
            pretty["dataset"] = pretty["dataset"]  # bereits sinnvoll benannt
        sort_col = "r2_mean"
        sel_cols = ["dataset", "model", "r2_mean", "mae_mean", "rmse_mean"]
        pretty = pretty[sel_cols]
        # Umbenennen für einheitliche Ausgabe
        pretty = pretty.rename(columns={
            "r2_mean": "best_r2",
            "mae_mean": "best_mae",
            "rmse_mean": "best_rmse"
        })
    else:
        raise ValueError(
            "Unbekanntes CSV-Schema. Erwartet entweder Spalten "
            "{best_r2,best_mae,best_rmse} oder {r2_mean,mae_mean,rmse_mean}."
        )

    pretty_sorted = pretty.sort_values("best_r2", ascending=False)
    top = pretty_sorted.head(top_n)

    # Ausgabeordner anlegen
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    top.to_csv(out_csv, index=False)

    print("\nTop-Kombinationen:")
    print(top.to_string(index=False))
    print(f"\nTop-{top_n} Kombinationen gespeichert unter: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", required=True, help="Pfad zur Vergleichs-/Summary-CSV")
    parser.add_argument("--out_csv", required=True, help="Zielpfad für die Top-Kandidaten")
    parser.add_argument("--top_n", type=int, default=10, help="Anzahl der Top-Kombinationen")
    args = parser.parse_args()

    select_top_candidates(args.summary_csv, args.out_csv, args.top_n)