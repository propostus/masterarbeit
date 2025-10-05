# scripts/compare_pycaret_all_datasets.py
"""
Vergleicht automatisch alle ausgewählten Feature-Datasets mit PyCaret.
Für jedes Dataset wird ein Regressionsvergleich (compare_models) durchgeführt.
Ergebnisse werden in einer zentralen CSV gespeichert.
"""

import os
import pandas as pd
from pycaret.regression import setup, compare_models, pull
from datetime import datetime

def run_pycaret_comparison(dataset_dir, out_csv, target_col="wer", random_state=42):
    # Alle Datasets aus dem angegebenen Verzeichnis sammeln
    csv_files = sorted([
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".csv") and "dataset_" in f
    ])

    if not csv_files:
        print(f"Keine Datasets in {dataset_dir} gefunden.")
        return

    all_results = []

    for csv_path in csv_files:
        print(f"\n=== Vergleiche Modelle für Dataset: {os.path.basename(csv_path)} ===")

        try:
            df = pd.read_csv(csv_path)

            # Sicherstellen, dass Zielspalte vorhanden ist
            if target_col not in df.columns:
                print(f"Überspringe {csv_path}, keine Spalte '{target_col}' gefunden.")
                continue

            # Setup von PyCaret (aktuelle API, kein 'silent' mehr)
            setup(
                data=df,
                target=target_col,
                session_id=random_state,
                normalize=True,
                normalize_method="zscore",
                imputation_type="simple",
                verbose=False,
                profile=False,   # kein interaktives Profiling
            )

            # Modellvergleich (standardmäßig Top-3 Modelle)
            best_model = compare_models(sort="R2", n_select=3)
            results_df = pull()

            results_df["dataset"] = os.path.basename(csv_path)
            results_df["timestamp"] = datetime.now().isoformat(timespec="seconds")
            all_results.append(results_df)

        except Exception as e:
            print(f"Fehler bei {csv_path}: {e}")

    if not all_results:
        print("Keine Ergebnisse generiert.")
        return

    # Alle Ergebnisse zusammenführen und speichern
    merged_results = pd.concat(all_results, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged_results.to_csv(out_csv, index=False)
    print(f"\nAlle Ergebnisse gespeichert in: {out_csv}")


if __name__ == "__main__":
    dataset_dir = "results/datasets/selected"
    out_csv = "results/compare/pycaret_model_comparison_full_std.csv"
    run_pycaret_comparison(dataset_dir, out_csv)