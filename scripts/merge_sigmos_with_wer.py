# scripts/merge_sigmos_with_wer.py

import os
import pandas as pd

def merge_sigmos_with_wer(sigmos_csv, wer_dir, output_dir):
    print("=== Führe Merge von SigMOS-Features mit WER-Ergebnissen durch ===")

    sigmos_df = pd.read_csv(sigmos_csv)
    print(f"Geladen: {sigmos_csv} ({sigmos_df.shape[0]} Zeilen)")

    # Einheitlicher Dateiname
    if "file" in sigmos_df.columns:
        sigmos_df = sigmos_df.rename(columns={"file": "filename"})
    sigmos_df["filename"] = sigmos_df["filename"].apply(os.path.basename)

    wer_variants = {
        "base": os.path.join(wer_dir, "wer_base_full.csv"),
        "small": os.path.join(wer_dir, "wer_small_full.csv"),
        "tiny": os.path.join(wer_dir, "wer_tiny_full.csv"),
    }

    os.makedirs(output_dir, exist_ok=True)

    for size, wer_path in wer_variants.items():
        if not os.path.exists(wer_path):
            print(f"Datei nicht gefunden: {wer_path}")
            continue

        wer_df = pd.read_csv(wer_path)

        # Einheitlicher Dateiname in WER
        if "file" in wer_df.columns:
            wer_df = wer_df.rename(columns={"file": "filename"})
        elif "path" in wer_df.columns:
            wer_df = wer_df.rename(columns={"path": "filename"})
        wer_df["filename"] = wer_df["filename"].apply(os.path.basename)

        # Mögliche WER-Spalten
        wer_cols_all = [c for c in wer_df.columns if "wer" in c.lower()]
        if not wer_cols_all:
            print(f"Keine WER-Spalten in {wer_path} gefunden.")
            continue

        # Bevorzugt bereits modell-spezifische Spalte, sonst generische 'wer'
        prefer = f"wer_{size}"
        if prefer in wer_df.columns:
            wer_col = prefer
        elif "wer" in wer_df.columns:
            wer_col = "wer"
        else:
            # Fallback: nimm die erste gefundene WER-Spalte
            wer_col = wer_cols_all[0]

        # Nach Merge ggf. umbenennen -> wer_base/wer_small/wer_tiny
        target_wer_col = f"wer_{size}"
        subset = wer_df[["filename", wer_col]].rename(columns={wer_col: target_wer_col})

        merged = sigmos_df.merge(subset, on="filename", how="inner")
        out_path = os.path.join(output_dir, f"embeddings_sigmos_{size}.csv")
        merged.to_csv(out_path, index=False)

        print(f"→ {size} fertig: {out_path} ({merged.shape[0]} Zeilen, {merged.shape[1]} Spalten)")

    print("Alle Varianten erfolgreich gemergt.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge SigMOS-Features mit WER-Ergebnissen (base/small/tiny)")
    parser.add_argument("--sigmos_csv", type=str, default="results/embeddings/sigmos_embeddings_full.csv")
    parser.add_argument("--wer_dir", type=str, default="results/wer")
    parser.add_argument("--output_dir", type=str, default="results/datasets/embeddings_merged")
    args = parser.parse_args()

    merge_sigmos_with_wer(args.sigmos_csv, args.wer_dir, args.output_dir)


if __name__ == "__main__":
    main()