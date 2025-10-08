# scripts/batch_optimize_models.py

import os
import subprocess
import argparse

def run_batch_optimization(dataset_dir, target_col, out_dir, models, transforms):
    """
    Führt optimize_transformed_target.py für alle CSV-Dateien in dataset_dir aus.
    Jede Datei wird separat mit denselben Modellen und Transformationen optimiert.
    """

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Ordner {dataset_dir} existiert nicht.")
    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"Keine CSV-Dateien in {dataset_dir} gefunden.")

    print(f"Starte Batch-Optimierung für {len(csv_files)} Dateien in {dataset_dir}")
    print(f"Zielspalte: {target_col}")
    print(f"Ausgabeordner: {out_dir}\n")

    for csv_name in csv_files:
        dataset_path = os.path.join(dataset_dir, csv_name)
        print(f"=== Verarbeite {csv_name} ===")

        cmd = [
            "python", "-m", "scripts.optimize_transformed_target",
            "--dataset_csv", dataset_path,
            "--target_col", target_col,
            "--models", *models,
            "--transforms", *transforms,
            "--out_dir", out_dir
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[FEHLER] Fehler bei {csv_name} – wird übersprungen.\n")
            continue

        print(f"Fertig: {csv_name}\n{'-'*80}\n")

    print("Batch-Optimierung abgeschlossen!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-Optimierung für alle Datasets in einem Ordner")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Ordner mit Datasets (CSV-Dateien)")
    parser.add_argument("--target_col", type=str, required=True, help="Zielspalte, z. B. wer_tiny, wer_base, wer_small")
    parser.add_argument("--out_dir", type=str, required=True, help="Ausgabeordner für die Ergebnisse")
    parser.add_argument("--models", nargs="+", default=["ridge", "lasso", "rf", "gbr", "hgb", "svr"],
                        help="Liste der Modelle")
    parser.add_argument("--transforms", nargs="+", default=["none", "yeojohnson", "log1p", "boxcox"],
                        help="Liste der Ziel-Transformationen")

    args = parser.parse_args()

    run_batch_optimization(
        dataset_dir=args.dataset_dir,
        target_col=args.target_col,
        out_dir=args.out_dir,
        models=args.models,
        transforms=args.transforms
    )