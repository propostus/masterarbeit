# scripts/merge_noisy_embeddings_with_wer.py
"""
Führt die noisy WER-Dateien mit den zugehörigen SigMOS- und WavLM-Embeddings zusammen.
Erstellt pro Whisper-Modell (tiny, base, small) eine kombinierte CSV mit allen SNR-Stufen.
"""

import os
import argparse
import pandas as pd


def merge_embeddings_with_wer(wer_dir, emb_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    whisper_models = ["tiny", "base", "small"]
    snr_levels = [0, 10, 20]

    for model in whisper_models:
        print(f"\n=== Verarbeite {model.upper()} ===")
        merged_frames = []

        for snr in snr_levels:
            wer_path = os.path.join(wer_dir, f"wer_{model}_snr{snr}.csv")
            sigmos_path = os.path.join(emb_dir, f"sigmos_embeddings_snr{snr}.csv")
            wavlm_path = os.path.join(emb_dir, f"wavlm_embeddings_snr{snr}.csv")

            if not os.path.exists(wer_path):
                print(f"Fehlt: {wer_path}")
                continue
            if not os.path.exists(sigmos_path):
                print(f"Fehlt: {sigmos_path}")
                continue
            if not os.path.exists(wavlm_path):
                print(f"Fehlt: {wavlm_path}")
                continue

            wer_df = pd.read_csv(wer_path)
            sig_df = pd.read_csv(sigmos_path)
            wav_df = pd.read_csv(wavlm_path)

            # Einheitliche Spaltennamen
            wer_df["filename"] = wer_df["filename"].astype(str).apply(os.path.basename)
            sig_df["filename"] = sig_df["file"].astype(str).apply(os.path.basename)
            wav_df["filename"] = wav_df["filename"].astype(str).apply(os.path.basename)

            # Merge SigMOS + WavLM + WER
            merged = (
                wer_df.merge(sig_df.drop(columns=["file"], errors="ignore"), on="filename", how="inner")
                      .merge(wav_df, on="filename", how="inner")
            )

            merged["snr"] = snr
            merged.rename(columns={"wer": f"wer_{model}"}, inplace=True)
            merged_frames.append(merged)

            print(f"   {model} SNR {snr}: {merged.shape[0]} Zeilen")

        if merged_frames:
            merged_all = pd.concat(merged_frames, ignore_index=True)
            out_path = os.path.join(out_dir, f"embeddings_sigmos_wavlm_noisy_{model}.csv")
            merged_all.to_csv(out_path, index=False)
            print(f"Gespeichert: {out_path} ({merged_all.shape[0]} Zeilen)")
        else:
            print(f"Keine Daten für Modell {model} gefunden.")


def main():
    parser = argparse.ArgumentParser(description="Merge noisy WER- und Embedding-Dateien pro Whisper-Modell")
    parser.add_argument("--wer_dir", type=str, required=True, help="Pfad zu results/wer/noisy/")
    parser.add_argument("--emb_dir", type=str, required=True, help="Pfad zu results/embeddings/noisy/")
    parser.add_argument("--out_dir", type=str, required=True, help="Pfad zum Zielverzeichnis (z. B. results/datasets/noisy/)")
    args = parser.parse_args()

    merge_embeddings_with_wer(args.wer_dir, args.emb_dir, args.out_dir)


if __name__ == "__main__":
    main()