# scripts/summarize_sigmos_results.py
import os
import pandas as pd

def summarize_sigmos_results(base_dir="results"):
    print("=== Zusammenfassung SigMOS-Optimierungen ===")

    subdirs = [
        ("sigmos_tiny", os.path.join(base_dir, "sigmos_optuna_tiny")),
        ("sigmos_small", os.path.join(base_dir, "sigmos_optuna_small")),
        ("sigmos_base", os.path.join(base_dir, "sigmos_optuna_base")),
    ]

    rows = []
    for name, path in subdirs:
        summary_path = os.path.join(path, "study_summary.csv")
        params_path = os.path.join(path, "best_params.json")
        if not os.path.exists(summary_path):
            print(f"Kein Ergebnis gefunden für {name} unter {summary_path}")
            continue

        df = pd.read_csv(summary_path)
        metrics = df.iloc[0].to_dict()
        metrics["model"] = name
        rows.append(metrics)

    if not rows:
        print("Keine Ergebnisse gefunden.")
        return

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df[["model", "r2", "rmse", "mae"]]

    out_path = os.path.join(base_dir, "sigmos_summary.csv")
    summary_df.to_csv(out_path, index=False)

    print("\nErgebnisse:")
    print(summary_df.to_string(index=False))
    print(f"\nGespeichert unter: {out_path}")


if __name__ == "__main__":
    summarize_sigmos_results()