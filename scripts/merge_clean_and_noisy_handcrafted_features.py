# scripts/merge_clean_and_noisy_handcrafted_features.py
import os
import argparse
import pandas as pd

def norm_name(x: str) -> str:
    return os.path.basename(str(x)).strip().lower()

def load_df(path: str, is_clean: bool):
    df = pd.read_csv(path)
    if "filename" not in df.columns:
        for alt in ["file", "path"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "filename"})
                break
    df["filename"] = df["filename"].map(norm_name)
    if "snr" not in df.columns:
        df["snr"] = "clean" if is_clean else "unknown"
    elif is_clean:
        df["snr"] = "clean"
    return df

def main():
    parser = argparse.ArgumentParser(description="Clean- und Noisy-Features zusammenführen und SNR kennzeichnen.")
    parser.add_argument("--clean_csv", required=True, help="Pfad zu features_dataset_full.csv (clean)")
    parser.add_argument("--noisy_csv", required=True, help="Pfad zu features_noisy_all.csv (noisy, mit snr 0/10/20)")
    parser.add_argument("--out_csv", required=True, help="Pfad zur Ausgabe-Datei")
    args = parser.parse_args()

    clean = load_df(args.clean_csv, is_clean=True)
    noisy = load_df(args.noisy_csv, is_clean=False)

    # gleiche Spaltenmenge
    common_cols = sorted(list(set(clean.columns) & set(noisy.columns)))
    clean = clean[common_cols]
    noisy = noisy[common_cols]

    combined = pd.concat([clean, noisy], ignore_index=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    combined.to_csv(args.out_csv, index=False)

    print(f"Gespeichert: {args.out_csv}")
    print(f"Zeilen: {len(combined)}, Spalten: {len(combined.columns)}")
    print(f"SNR-Werte im Datensatz: {combined['snr'].unique()}")

if __name__ == "__main__":
    main()