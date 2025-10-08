# scripts/batch_optimize_tabular.py

import os
import subprocess
import argparse


def run_batch_tabular(dataset_dir, target_col, out_dir, models):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Ordner {dataset_dir} existiert nicht.")
    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".csv")])

    print(f"Starte Batch-Optimierung für {len(csv_files)} Dateien in {dataset_dir}")
    print(f"Zielspalte: {target_col}")
    print(f"Ausgabeordner: {out_dir}\n")

    for csv_name in csv_files:
        dataset_path = os.path.join(dataset_dir, csv_name)
        print(f"=== Verarbeite {csv_name} ===")

        cmd = [
            "python", "-m", "scripts.optimize_tabular_models",
            "--dataset_csv", dataset_path,
            "--target_col", target_col,
            "--models", *models,
            "--out_dir", out_dir
        ]

        subprocess.run(cmd, check=False)
        print(f"Fertig: {csv_name}\n{'-'*80}\n")

    print("Batch-Optimierung abgeschlossen!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-Optimierung für Tabular-Modelle (LightGBM, CatBoost)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Ordner mit Feature-Datasets (CSV)")
    parser.add_argument("--target_col", type=str, required=True, help="Zielspalte, z. B. wer_tiny, wer_base, wer_small")
    parser.add_argument("--out_dir", type=str, required=True, help="Ausgabeordner für Ergebnisse")
    parser.add_argument("--models", nargs="+", default=["lightgbm", "catboost"], help="Zu optimierende Modelle")
    args = parser.parse_args()

    run_batch_tabular(args.dataset_dir, args.target_col, args.out_dir, args.models)