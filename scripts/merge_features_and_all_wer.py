# scripts/merge_features_and_all_wer.py
import os
import argparse
import pandas as pd

def normalize_name(s: str) -> str:
    s = str(s).strip()
    s = os.path.basename(s)
    return s.lower()

def merge_features_and_all_wer(features_csv, wer_dir, out_csv, debug_dir="results/datasets/_debug"):
    print(f"Lade Features: {features_csv}")
    df_features = pd.read_csv(features_csv)
    df_features["__key"] = df_features["filename"].map(normalize_name)

    # Finde alle WER-Dateien
    wer_files = [
        os.path.join(wer_dir, f)
        for f in os.listdir(wer_dir)
        if f.endswith(".csv") and "wer_" in f
    ]

    if not wer_files:
        print(f"Keine WER-Dateien in {wer_dir} gefunden.")
        return

    df_merged = df_features.copy()
    for wer_path in wer_files:
        wer_name = os.path.basename(wer_path).replace(".csv", "")
        wer_label = wer_name.replace("wer_", "").replace("_full", "")

        print(f"→ Füge hinzu: {wer_label}")
        df_wer = pd.read_csv(wer_path)
        df_wer["__key"] = df_wer["filename"].map(normalize_name)

        if "wer" not in df_wer.columns:
            print(f"Datei {wer_name} hat keine Spalte 'wer' – überspringe.")
            continue

        df_wer = df_wer[["__key", "wer"]].rename(columns={"wer": f"wer_{wer_label}"})
        df_merged = pd.merge(df_merged, df_wer, on="__key", how="left")

    # Aufräumen
    df_merged.drop(columns=["__key"], inplace=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    df_merged.to_csv(out_csv, index=False)
    print(f"\nGesamtes Merged Dataset gespeichert unter: {out_csv}")
    print(f"Shape: {df_merged.shape}")
    print(f"WER-Spalten: {[c for c in df_merged.columns if c.startswith('wer_')]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--wer_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    merge_features_and_all_wer(args.features_csv, args.wer_dir, args.out_csv)