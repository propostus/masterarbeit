# scripts/select_top20_all.py

import os
import pandas as pd

def collect_results(base_dir, label):
    """Lädt alle summary_*.csv aus einem Ergebnisordner."""
    files = [f for f in os.listdir(base_dir) if f.startswith("summary_") and f.endswith(".csv")]
    dfs = []
    for f in files:
        path = os.path.join(base_dir, f)
        try:
            df = pd.read_csv(path)
            df["whisper_size"] = label
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Fehler beim Laden von {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def select_top(df, n=20):
    """Gibt die Top-n Zeilen nach best_r2 zurück."""
    df_sorted = df.sort_values("best_r2", ascending=False)
    return df_sorted.head(n)

def main():
    results = []

    configs = [
        ("results/optimized_tiny_full_models", "tiny"),
        ("results/optimized_base_full_models", "base"),
        ("results/optimized_small_full_models", "small"),
    ]

    for folder, label in configs:
        if not os.path.exists(folder):
            print(f"[WARN] {folder} nicht gefunden, wird übersprungen.")
            continue

        df = collect_results(folder, label)
        if df.empty:
            print(f"[WARN] Keine Ergebnisse in {folder}.")
            continue

        top = select_top(df, 20)
        print(f"\n=== Top 20 für {label.upper()} ===")
        print(top[["dataset", "model", "transform", "best_r2", "best_mae", "best_rmse"]])
        results.append(top)

    if results:
        combined = pd.concat(results, ignore_index=True)
        os.makedirs("results/compare", exist_ok=True)
        combined.to_csv("results/compare/top20_all_models.csv", index=False)
        print("\nZusammenfassung gespeichert unter: results/compare/top20_all_models.csv")
    else:
        print("Keine Daten gefunden.")

if __name__ == "__main__":
    main()