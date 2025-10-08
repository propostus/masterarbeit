# scripts/summarize_optimizations.py
"""
Fasst alle Optimierungsergebnisse (opt_*.csv) zusammen
und extrahiert je Dataset + Modell den besten Score.
"""

import os
import argparse
import pandas as pd


def summarize_optimizations(opt_dir, out_csv):
    # Alle opt_* CSV-Dateien im Verzeichnis finden
    opt_files = [
        os.path.join(opt_dir, f)
        for f in sorted(os.listdir(opt_dir))
        if f.startswith("opt_") and f.endswith(".csv")
    ]

    if not opt_files:
        print(f"Keine opt_*.csv Dateien in {opt_dir} gefunden.")
        return

    all_results = []

    for fpath in opt_files:
        df = pd.read_csv(fpath)
        if "mean_test_r2" not in df.columns:
            print(f"Überspringe {fpath} (keine Spalte mean_test_r2).")
            continue

        best_row = df.loc[df["mean_test_r2"].idxmax()]
        all_results.append({
            "file": os.path.basename(fpath),
            "dataset": os.path.basename(fpath)
                        .replace("opt_", "")
                        .replace(".csv", "")
                        .replace("results_datasets_selected_tiny_full_", ""),
            "model": best_row.get("param_model__alpha", None) or "ridge" if "ridge" in fpath else "hgb",
            "best_r2": best_row["mean_test_r2"],
            "best_mae": best_row.get("mean_test_mae", None),
            "best_rmse": best_row.get("mean_test_rmse", None),
            "best_params": str({k: best_row[k] for k in df.columns if k.startswith("param_")})
        })

    summary_df = pd.DataFrame(all_results)
    summary_df.sort_values("best_r2", ascending=False, inplace=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\nZusammenfassung gespeichert unter: {out_csv}")
    print(summary_df.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt_dir", required=True, help="Ordner mit opt_*.csv Dateien")
    parser.add_argument("--out_csv", required=True, help="Output-Datei für Zusammenfassung")
    args = parser.parse_args()

    summarize_optimizations(args.opt_dir, args.out_csv)