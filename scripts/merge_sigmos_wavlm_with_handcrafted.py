import os
import argparse
import pandas as pd

def normalize_filename(x: str) -> str:
    return os.path.basename(str(x)).strip().lower()

def load_df(path: str, label: str):
    df = pd.read_csv(path)
    if "filename" not in df.columns:
        for alt in ["file", "path"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "filename"})
                break
    df["filename"] = df["filename"].map(normalize_filename)
    if "snr" not in df.columns:
        df["snr"] = "clean"
    df["source"] = label
    return df

def merge_datasets(sigmos_wavlm_csv, handcrafted_csv, out_csv):
    print(f"Lade SigMOS+WavLM: {sigmos_wavlm_csv}")
    df_embed = load_df(sigmos_wavlm_csv, label="embeddings")

    print(f"Lade Handcrafted Features: {handcrafted_csv}")
    df_hand = load_df(handcrafted_csv, label="handcrafted")

    # Gleiche 'snr'-Typen sicherstellen
    df_embed["snr"] = df_embed["snr"].astype(str)
    df_hand["snr"] = df_hand["snr"].astype(str)

    # Merge auf Basis von (filename, snr)
    merged = pd.merge(df_embed, df_hand, on=["filename", "snr"], how="inner", suffixes=("_embed", "_hand"))

    print(f"Gemergte Zeilen: {len(merged)} (von {len(df_embed)} Embeddings, {len(df_hand)} Handcrafted)")
    print(f"SNR-Werte im Merge: {merged['snr'].unique()}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"Gespeichert unter: {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Merge SigMOS+WavLM mit Handcrafted Features")
    parser.add_argument("--sigmos_wavlm_csv", required=True, help="Pfad zu embeddings_sigmos_wavlm_clean_and_noisy_XXX.csv")
    parser.add_argument("--handcrafted_csv", required=True, help="Pfad zu features_clean_and_noisy.csv")
    parser.add_argument("--out_csv", required=True, help="Pfad zur Ausgabe")
    args = parser.parse_args()

    merge_datasets(args.sigmos_wavlm_csv, args.handcrafted_csv, args.out_csv)

if __name__ == "__main__":
    main()