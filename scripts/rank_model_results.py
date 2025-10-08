# scripts/rank_model_results.py
import os
import argparse
import pandas as pd

def rank_models(csv_path, out_path):
    df = pd.read_csv(csv_path)

    # Sauberstellen: nur relevante Spalten
    required_cols = {"model", "dataset", "r2_mean", "mae_mean", "rmse_mean"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Fehlende Spalten: {required_cols - set(df.columns)}")

    # Durchschnitts-Ranking nach allen Metriken
    df["rank_r2"] = df["r2_mean"].rank(ascending=False)
    df["rank_mae"] = df["mae_mean"].rank(ascending=True)
    df["rank_rmse"] = df["rmse_mean"].rank(ascending=True)
    df["rank_avg"] = df[["rank_r2", "rank_mae", "rank_rmse"]].mean(axis=1)

    # Aggregiertes Modell-Ranking
    summary = (
        df.groupby("model")[["r2_mean", "mae_mean", "rmse_mean", "rank_avg"]]
        .mean()
        .sort_values("rank_avg")
        .reset_index()
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Ranking gespeichert unter: {out_path}")
    print(summary.head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Pfad zu compare_*.csv")
    parser.add_argument("--out_csv", required=True, help="Pfad für das Ranking-CSV")
    args = parser.parse_args()

    rank_models(args.csv, args.out_csv)