# scripts/merge_augmented_embeddings_and_wer.py
# --------------------------------------------
# Merged SigMOS + WavLM embeddings mit WER-Ergebnissen
# für augmentierte Dateien (ESC-50 noisy)
# --------------------------------------------

import os
import argparse
import pandas as pd

def normalize_filename_column(df):
    """Findet Spalte 'file' oder 'filename' und gibt vereinheitlicht 'filename' zurück."""
    cols = [c.lower() for c in df.columns]
    if "file" in cols:
        df = df.rename(columns={df.columns[cols.index("file")]: "filename"})
    elif "filename" in cols:
        df = df.rename(columns={df.columns[cols.index("filename")]: "filename"})
    else:
        raise ValueError("Keine Spalte 'file' oder 'filename' gefunden.")
    df["filename"] = df["filename"].astype(str).str.lower()
    return df

def merge_embeddings_and_wer(sigmos_csv, wavlm_csv, wer_dir, out_csv):
    print("=== Merging Embeddings + WER (Augmented) ===")

    # Lade Embeddings
    df_sigmos = pd.read_csv(sigmos_csv)
    df_sigmos = normalize_filename_column(df_sigmos)

    df_wavlm = pd.read_csv(wavlm_csv)
    df_wavlm = normalize_filename_column(df_wavlm)

    # Merge Embeddings
    df_emb = pd.merge(df_sigmos, df_wavlm, on="filename", suffixes=("_sigmos", "_wavlm"))
    print(f"→ Embeddings gemerged: {df_emb.shape}")

    # WER-Dateien
    wer_files = {
        "tiny": os.path.join(wer_dir, "wer_tiny_esc50_snr15_30.csv"),
        "base": os.path.join(wer_dir, "wer_base_esc50_snr15_30.csv"),
        "small": os.path.join(wer_dir, "wer_small_esc50_snr15_30.csv"),
    }

    for size, path in wer_files.items():
        if os.path.exists(path):
            df_wer = pd.read_csv(path)
            df_wer = normalize_filename_column(df_wer)

            df_emb = pd.merge(
                df_emb,
                df_wer[["filename", "wer"]].rename(columns={"wer": f"wer_{size}"}),
                on="filename",
                how="left",
            )
            print(f"✓ {size} WER gemerged ({df_wer.shape[0]} Zeilen)")
        else:
            print(f"⚠️ Datei fehlt: {path}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_emb.to_csv(out_csv, index=False)
    print(f"\n✅ Ergebnis gespeichert unter: {out_csv}")
    print(f"Endgültige Form: {df_emb.shape}")

def main():
    parser = argparse.ArgumentParser(description="Merge SigMOS + WavLM embeddings mit WER-Ergebnissen (Augmented)")
    parser.add_argument("--sigmos_csv", required=True, help="Pfad zu SigMOS-Embeddings (CSV)")
    parser.add_argument("--wavlm_csv", required=True, help="Pfad zu WavLM-Embeddings (CSV)")
    parser.add_argument("--wer_dir", required=True, help="Ordner mit WER-Dateien (tiny/base/small)")
    parser.add_argument("--out_csv", required=True, help="Pfad zur kombinierten Ausgabedatei (CSV)")
    args = parser.parse_args()

    merge_embeddings_and_wer(args.sigmos_csv, args.wavlm_csv, args.wer_dir, args.out_csv)

if __name__ == "__main__":
    main()