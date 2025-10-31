# scripts/merge_wer_unseen.py
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--small", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    df_tiny = pd.read_csv(args.tiny).rename(columns={"wer": "wer_tiny"})
    df_base = pd.read_csv(args.base).rename(columns={"wer": "wer_base"})
    df_small = pd.read_csv(args.small).rename(columns={"wer": "wer_small"})

    # Einheitliche Spalte "filename"
    for df in [df_tiny, df_base, df_small]:
        df["filename"] = df["filename"].apply(lambda x: os.path.basename(str(x)).lower())

    # Mergen über filename
    merged = df_tiny.merge(df_base[["filename", "wer_base"]], on="filename", how="outer")
    merged = merged.merge(df_small[["filename", "wer_small"]], on="filename", how="outer")

    print(f"Zusammengeführt: {len(merged)} Zeilen")
    merged.to_csv(args.out_csv, index=False)
    print(f"Gespeichert unter: {args.out_csv}")

if __name__ == "__main__":
    main()