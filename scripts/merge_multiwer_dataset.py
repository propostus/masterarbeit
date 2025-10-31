# scripts/merge_multiwer_dataset.py

import pandas as pd
import os

def main():
    # Eingabepfade (aus deinem setup)
    base_dir = "results/datasets/clean_and_noisy"
    tiny_path = os.path.join(base_dir, "embeddings_sigmos_wavlm_clean_and_noisy_tiny.csv")
    base_path = os.path.join(base_dir, "embeddings_sigmos_wavlm_clean_and_noisy_base.csv")
    small_path = os.path.join(base_dir, "embeddings_sigmos_wavlm_clean_and_noisy_small.csv")

    # Zielpfad
    out_path = "results/datasets/merged_sigmos_wavlm_multiwer.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # CSVs laden
    print("Lade Datensätze ...")
    tiny = pd.read_csv(tiny_path)
    base = pd.read_csv(base_path)
    small = pd.read_csv(small_path)

    # Embedding-Spalten identifizieren
    embed_cols = [c for c in tiny.columns if c.startswith("embed_")]
    print(f"Gefundene Embedding-Spalten: {len(embed_cols)}")

    # Merge anhand von filename UND snr
    merged = (
        tiny[["filename", "snr"] + embed_cols + ["wer_tiny"]]
        .merge(base[["filename", "snr", "wer_base"]], on=["filename", "snr"])
        .merge(small[["filename", "snr", "wer_small"]], on=["filename", "snr"])
    )

    # Ausgabe
    print(f"Shape des gemergten Datensatzes: {merged.shape}")
    print("Beispielspalten:", list(merged.columns)[:15])

    # Datei speichern
    merged.to_csv(out_path, index=False)
    print(f"Gespeichert unter: {out_path}")

    # Korrelation prüfen
    corr = merged[["wer_tiny", "wer_base", "wer_small"]].corr()
    print("\nKorrelation zwischen den Zielgrößen:")
    print(corr)

    corr_out_path = "results/datasets/multiwer_correlations.csv"
    corr.to_csv(corr_out_path)
    print(f"Korrelation gespeichert unter: {corr_out_path}")


if __name__ == "__main__":
    main()