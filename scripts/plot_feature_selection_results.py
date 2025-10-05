import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_selection_results(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Farbcodierung nach Strategie
    palette = {
        "rf": "#1f77b4",
        "corr": "#ff7f0e",
        "mi": "#2ca02c",
        "pca": "#9467bd"
    }

    metrics = [
        ("r2_mean", "R² (mean)", "feature_selection_r2.png"),
        ("mae_mean", "MAE (mean)", "feature_selection_mae.png"),
        ("rmse_mean", "RMSE (mean)", "feature_selection_rmse.png"),
    ]

    for col, ylabel, filename in metrics:
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df,
            x="features",
            y=col,
            hue="strategy",
            marker="o",
            palette=palette
        )
        plt.title(f"{ylabel} vs. Feature-Anzahl")
        plt.xlabel("Anzahl Features")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
        plt.close()

    print(f"Plots gespeichert in: {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    plot_feature_selection_results(args.csv_path, args.out_dir)