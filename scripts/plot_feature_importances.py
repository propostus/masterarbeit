# scripts/plot_feature_importances.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_feature_importances(csv_path, out_png, top_n=20):
    # CSV einlesen
    df = pd.read_csv(csv_path, header=None, names=["feature", "importance"])
    
    # sortieren nach Wichtigkeit
    df_sorted = df.sort_values("importance", ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.barh(df_sorted["feature"][::-1], df_sorted["importance"][::-1])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances (RandomForest)")
    plt.tight_layout()
    
    # speichern
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    print(f"Plot gespeichert unter: {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Pfad zu feature_importances.csv")
    parser.add_argument("--out_png", type=str, required=True, help="Pfad zum Ausgabe-PNG")
    parser.add_argument("--top_n", type=int, default=20, help="Anzahl der Top-Features (default=20)")
    args = parser.parse_args()

    plot_feature_importances(args.csv_path, args.out_png, args.top_n)