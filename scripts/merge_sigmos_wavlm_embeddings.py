# scripts/merge_sigmos_wavlm_embeddings.py
import os
import pandas as pd

def merge_sigmos_and_wavlm(base_dir="results"):
    print("=== Merge: SigMOS + WavLM Embeddings ===")

    datasets_dir = os.path.join(base_dir, "datasets/embeddings_merged")
    out_dir = datasets_dir

    variants = ["tiny", "small", "base"]
    for variant in variants:
        sigmos_path = os.path.join(datasets_dir, f"embeddings_sigmos_{variant}.csv")
        wavlm_path = os.path.join(datasets_dir, f"embeddings_wavlm_{variant}.csv")
        out_path = os.path.join(out_dir, f"embeddings_sigmos_wavlm_{variant}.csv")

        if not os.path.exists(sigmos_path):
            print(f"Datei fehlt: {sigmos_path}")
            continue
        if not os.path.exists(wavlm_path):
            print(f"Datei fehlt: {wavlm_path}")
            continue

        print(f"\nMerging {variant}...")
        sig_df = pd.read_csv(sigmos_path)
        wav_df = pd.read_csv(wavlm_path)

        merged_df = pd.merge(sig_df, wav_df, on="filename", suffixes=("_sigmos", "_wavlm"))
        print(f"→ Vorbereitet: {merged_df.shape}")

        # WER-Spalte vereinheitlichen
        wer_sig = f"wer_{variant}_sigmos"
        wer_wav = f"wer_{variant}_wavlm"
        wer_final = f"wer_{variant}"

        if wer_sig in merged_df.columns:
            merged_df[wer_final] = merged_df[wer_sig]
        elif wer_wav in merged_df.columns:
            merged_df[wer_final] = merged_df[wer_wav]
        else:
            print(f"Warnung: Keine WER-Spalte gefunden für {variant}")
            continue

        # Alte WER-Spalten entfernen
        merged_df.drop(columns=[c for c in merged_df.columns if c.startswith("wer_") and c != wer_final], inplace=True)

        print(f"→ Nach Bereinigung: {merged_df.shape}")
        merged_df.to_csv(out_path, index=False)
        print(f"Gespeichert unter: {out_path}")

    print("\nMerge aller Varianten abgeschlossen.")


if __name__ == "__main__":
    merge_sigmos_and_wavlm()