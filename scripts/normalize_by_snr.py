# scripts/normalize_by_snr.py
# Normalisiert alle numerischen Feature-Spalten pro SNR-Stufe (Z-Score-Normalisierung)
# Ergebnis: *_normalized.csv wird erzeugt

import argparse
import pandas as pd
import numpy as np
import os


def normalize_by_snr(df: pd.DataFrame, group_col: str = "snr") -> pd.DataFrame:
    """
    Führt Z-Score-Normalisierung (pro SNR-Gruppe) für alle numerischen Features durch.
    """
    df_norm = df.copy()
    numeric_cols = [
        c for c in df.columns
        if np.issubdtype(df[c].dtype, np.number)
        and c not in ["wer_tiny", "wer_base", "wer_small"]
    ]

    grouped = df.groupby(group_col)
    for name, group in grouped:
        mean = group[numeric_cols].mean()
        std = group[numeric_cols].std(ddof=0).replace(0, 1.0)  # Schutz vor std=0
        df_norm.loc[group.index, numeric_cols] = (group[numeric_cols] - mean) / std
        print(f"→ Normalisiert Gruppe '{name}' ({len(group)} Zeilen)")

    return df_norm


def main():
    parser = argparse.ArgumentParser(description="Normalisiert Features pro SNR-Gruppe (Z-Score)")
    parser.add_argument("--dataset", required=True, help="Pfad zur Original-CSV (z. B. embeddings_sigmos_wavlm_clean_and_noisy_tiny.csv)")
    parser.add_argument("--out_path", type=str, default=None, help="Pfad zur Ausgabe-CSV (Standard: *_normalized.csv)")
    args = parser.parse_args()

    # === Laden ===
    print(f"=== Lade Dataset: {args.dataset} ===")
    df = pd.read_csv(args.dataset, low_memory=False)
    print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    if "snr" not in df.columns:
        raise ValueError("Spalte 'snr' fehlt — erforderlich für Normalisierung pro SNR-Gruppe!")

    # === Normalisierung ===
    df_norm = normalize_by_snr(df, group_col="snr")

    # === Speichern ===
    out_path = args.out_path or args.dataset.replace(".csv", "_normalized.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_norm.to_csv(out_path, index=False)
    print(f"\nNormalisiertes Dataset gespeichert unter: {out_path}")


if __name__ == "__main__":
    main()