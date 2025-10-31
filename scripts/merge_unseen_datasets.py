# scripts/merge_unseen_datasets.py
import pandas as pd
import os
import argparse

def normalize_filename(x: str):
    """Normiert Dateinamen (lowercase + basename)."""
    return os.path.basename(str(x)).lower().strip()

def main():
    parser = argparse.ArgumentParser(description="Merge unseen WER + Embeddings (SigMOS + WavLM).")
    parser.add_argument("--sigmos", required=True)
    parser.add_argument("--wavlm", required=True)
    parser.add_argument("--wer", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    # === Dateien laden ===
    print("Lade SigMOS Embeddings...")
    df_sig = pd.read_csv(args.sigmos)
    print(f"SigMOS: {df_sig.shape}")

    print("Lade WavLM Embeddings...")
    df_wav = pd.read_csv(args.wavlm)
    print(f"WavLM: {df_wav.shape}")

    print("Lade WER-Dateien...")
    df_wer = pd.read_csv(args.wer)
    print(f"WER: {df_wer.shape}")

    # === Spalten vereinheitlichen ===
    if "file" in df_sig.columns:
        df_sig = df_sig.rename(columns={"file": "filename"})
    if "filename" not in df_sig.columns:
        raise ValueError("SigMOS-Datei enthält keine Spalte 'filename' oder 'file'.")

    if "filename" not in df_wav.columns:
        raise ValueError("WavLM-Datei enthält keine Spalte 'filename'.")

    # === Normalisierung der Filenamen ===
    for df in [df_sig, df_wav, df_wer]:
        df["filename"] = df["filename"].apply(normalize_filename)

    # === Mergen ===
    merged = df_wer.merge(df_sig, on="filename", how="inner")
    merged = merged.merge(df_wav, on="filename", how="inner")

    print(f"\nErfolgreich gemerged: {merged.shape[0]} Zeilen, {merged.shape[1]} Spalten")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Gespeichert unter: {args.out_csv}")

if __name__ == "__main__":
    main()