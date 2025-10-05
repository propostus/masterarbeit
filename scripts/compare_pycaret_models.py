# scripts/compare_pycaret_models.py
"""
Vergleicht automatisch alle Regressionsmodelle mit PyCaret über mehrere Datasets.
Ergebnisse (R2, RMSE, MAE etc.) werden in einer CSV-Datei zusammengefasst.
"""

import os
import argparse
import pandas as pd
from pycaret.regression import setup, compare_models, pull


def evaluate_datasets(dataset_paths, out_csv):
    """
    Führt PyCaret-Vergleiche über mehrere Datasets durch und speichert die Ergebnisse.
    """
    results = []

    for path in dataset_paths:
        print(f"\n==> Evaluating dataset: {path}")
        try:
            df = pd.read_csv(path)

            if "wer" not in df.columns:
                print(f"  'wer' column not found in {path}, skipping.")
                continue

            # Fehlende Werte im Ziel entfernen
            df = df.dropna(subset=["wer"])

            # Setup von PyCaret
            setup(
                data=df,
                target="wer",
                session_id=42,
                normalize=True,
                transformation=False,
                imputation_type="simple",
                silent=True,
                verbose=False,
                html=False,
            )

            # Alle Modelle vergleichen
            best_model = compare_models(sort="R2")
            comparison_df = pull()
            comparison_df["dataset"] = os.path.basename(path)

            results.append(comparison_df)

            print(f"  Finished {path}, best model: {comparison_df.iloc[0]['Model']}")

        except Exception as e:
            print(f"  Fehler bei {path}: {e}")

    # Ergebnisse zusammenführen und speichern
    if results:
        all_results = pd.concat(results, ignore_index=True)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        all_results.to_csv(out_csv, index=False)
        print(f"\nGesamtergebnisse gespeichert unter: {out_csv}")
    else:
        print("Keine gültigen Ergebnisse erzeugt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vergleich von PyCaret-Regressionsmodellen über mehrere Datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Liste der Dataset-Dateien (CSV).",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Pfad zur Ergebnis-CSV.",
    )
    args = parser.parse_args()

    evaluate_datasets(args.datasets, args.out_csv)