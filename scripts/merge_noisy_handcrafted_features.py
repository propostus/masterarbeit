# scripts/merge_noisy_handcrafted_features.py
import os
import argparse
import pandas as pd

def norm_name(x: str) -> str:
    return os.path.basename(str(x)).strip().lower()

def load_with_snr(path: str, snr_value: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "filename" not in df.columns:
        # Fallback: häufig wird die Spalte 'file' oder 'path' genannt
        for c in ["file", "path"]:
            if c in df.columns:
                df = df.rename(columns={c: "filename"})
                break
    df["filename"] = df["filename"].map(norm_name)
    df["snr"] = snr_value
    return df

def main():
    ap = argparse.ArgumentParser(description="Merge handcrafted features für SNR 0/10/20 in eine CSV")
    ap.add_argument("--snr0_csv", required=True)
    ap.add_argument("--snr10_csv", required=True)
    ap.add_argument("--snr20_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    d0 = load_with_snr(args.snr0_csv, "0")
    d1 = load_with_snr(args.snr10_csv, "10")
    d2 = load_with_snr(args.snr20_csv, "20")

    # vereinheitlichen: gleiche Spaltenmenge
    common_cols = set(d0.columns) & set(d1.columns) & set(d2.columns)
    d0 = d0[list(common_cols)]
    d1 = d1[list(common_cols)]
    d2 = d2[list(common_cols)]

    out = pd.concat([d0, d1, d2], ignore_index=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Gespeichert: {args.out_csv}  | Zeilen: {len(out)}  | Spalten: {len(out.columns)}")

if __name__ == "__main__":
    main()