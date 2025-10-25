# scripts/merge_clean_and_noisy_datasets.py
"""
Führt die cleanen und noisy Embedding-Datasets (SigMOS+WavLM) pro Whisper-Modell zusammen.
Verwendet ausschließlich die kombinierten SigMOS+WavLM-Dateien und entfernt Textspalten.
"""

import os
import argparse
import pandas as pd


def merge_clean_and_noisy(clean_dir, noisy_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    whisper_models = ["tiny", "base", "small"]

    for model in whisper_models:
        print(f"\n=== Merge clean + noisy für {model.upper()} ===")

        clean_path = os.path.join(clean_dir, f"embeddings_sigmos_wavlm_{model}.csv")
        noisy_path = os.path.join(noisy_dir, f"embeddings_sigmos_wavlm_noisy_{model}.csv")

        if not os.path.exists(clean_path):
            print(f"Clean-Datei fehlt: {clean_path}")
            continue
        if not os.path.exists(noisy_path):
            print(f"Noisy-Datei fehlt: {noisy_path}")
            continue

        clean_df = pd.read_csv(clean_path)
        noisy_df = pd.read_csv(noisy_path)

        # Einheitliche Struktur
        clean_df["snr"] = "clean"
        if "snr" not in noisy_df.columns:
            noisy_df["snr"] = "unknown"

        # Zusammenführen
        merged_df = pd.concat([clean_df, noisy_df], ignore_index=True)

        # Entferne irrelevante Textspalten, falls vorhanden
        drop_cols = [c for c in ["reference", "hypothesis"] if c in merged_df.columns]
        if drop_cols:
            merged_df = merged_df.drop(columns=drop_cols)
            print(f"Entfernte Spalten: {', '.join(drop_cols)}")

        out_path = os.path.join(out_dir, f"embeddings_sigmos_wavlm_clean_and_noisy_{model}.csv")
        merged_df.to_csv(out_path, index=False)

        print(f"Gespeichert: {out_path} ({merged_df.shape[0]} Zeilen, {merged_df.shape[1]} Spalten)")


def main():
    parser = argparse.ArgumentParser(description="Merge clean und noisy SigMOS+WavLM Datasets pro Whisper-Modell")
    parser.add_argument("--clean_dir", type=str, required=True, help="Pfad zu results/datasets/embeddings_merged/")
    parser.add_argument("--noisy_dir", type=str, required=True, help="Pfad zu results/datasets/noisy/")
    parser.add_argument("--out_dir", type=str, required=True, help="Pfad zu results/datasets/clean_and_noisy/")
    args = parser.parse_args()

    merge_clean_and_noisy(args.clean_dir, args.noisy_dir, args.out_dir)


if __name__ == "__main__":
    main()