# scripts/merge_embeddings_with_wer.py

import os
import argparse
import pandas as pd

def merge_embeddings_with_wer(embeddings_csv, wer_csv, out_csv):
    print(f"Lese Embeddings: {embeddings_csv}")
    emb = pd.read_csv(embeddings_csv)
    print(f"→ {emb.shape[0]} Samples, {emb.shape[1]-1} Embedding-Features")

    print(f"Lese WER-Datei: {wer_csv}")
    wer = pd.read_csv(wer_csv)
    print(f"→ {wer.shape[0]} WER-Zeilen")

    # Nur gemeinsame Samples (per Filename)
    merged = pd.merge(emb, wer[["filename", "wer"]], on="filename", how="inner")

    print(f"Nach Merge: {merged.shape[0]} gemeinsame Dateien")

    # Zielspaltenname anpassen
    model_name = os.path.basename(wer_csv).split("_")[1]  # z. B. 'base' aus 'wer_base_full.csv'
    merged = merged.rename(columns={"wer": f"wer_{model_name}"})

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"Kombinierte Datei gespeichert unter: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_csv", type=str, required=True)
    parser.add_argument("--wer_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    merge_embeddings_with_wer(args.embeddings_csv, args.wer_csv, args.out_csv)