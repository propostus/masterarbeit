# scripts/merge_features_and_wer.py
import os
import argparse
import pandas as pd

def normalize_name(s: str) -> str:
    s = str(s).strip()
    s = os.path.basename(s)
    return s.lower()

def merge_features_and_wer(features_csv, wer_csv, out_csv, debug_dir="results/datasets/_debug"):
    df_features = pd.read_csv(features_csv)
    df_wer = pd.read_csv(wer_csv)

    if "filename" not in df_features.columns or "filename" not in df_wer.columns:
        raise ValueError("Beide Dateien benötigen eine Spalte 'filename'.")

    # Normalisierte Keys
    df_features["__key"] = df_features["filename"].map(normalize_name)
    df_wer["__key"] = df_wer["filename"].map(normalize_name)

    # Merge
    df_merged = pd.merge(df_features, df_wer, on="__key", how="inner", suffixes=("_feat", "_wer"))

    # Bevorzugt den Originalnamen aus Features
    if "filename_feat" in df_merged.columns:
        df_merged.rename(columns={"filename_feat": "filename"}, inplace=True)
    if "filename_wer" in df_merged.columns:
        df_merged.drop(columns=["filename_wer"], inplace=True)

    # Hilfsspalte entfernen
    df_merged.drop(columns=["__key"], inplace=True)

    # Debug-Infos
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    feat_keys = set(df_features["__key"])
    wer_keys = set(df_wer["__key"])
    inter = feat_keys & wer_keys
    only_feat = sorted(list(feat_keys - wer_keys))
    only_wer = sorted(list(wer_keys - feat_keys))

    pd.Series(only_feat).to_csv(os.path.join(debug_dir, "missing_in_wer.csv"), index=False, header=False)
    pd.Series(only_wer).to_csv(os.path.join(debug_dir, "missing_in_features.csv"), index=False, header=False)

    # Speichern
    df_merged.to_csv(out_csv, index=False)

    print(f"Features: {df_features.shape}, WER: {df_wer.shape}")
    print(f"Gemeinsame Dateien: {len(inter)}")
    print(f"Nur in Features (Liste in {debug_dir}/missing_in_wer.csv): {len(only_feat)}")
    print(f"Nur in WER      (Liste in {debug_dir}/missing_in_features.csv): {len(only_wer)}")
    print(f"Merged Dataset gespeichert unter: {out_csv}")
    print(f"Endgültige Größe: {df_merged.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--wer_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    merge_features_and_wer(args.features_csv, args.wer_csv, args.out_csv)