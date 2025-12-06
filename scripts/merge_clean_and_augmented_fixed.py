# scripts/merge_clean_and_augmented_fixed.py
# ------------------------------------------------------
# Kombiniert clean + augmented Datasets mit robustem Matching
# (Berücksichtigt leicht unterschiedliche Filenamen wie "_aug" oder ".wav"/".mp3")
# ------------------------------------------------------

import os
import argparse
import pandas as pd
import re


def normalize_filename(s: str) -> str:
    """Normiert Dateinamen, entfernt Endungen und '_aug'-Suffix."""
    s = str(s).lower().strip()
    s = os.path.basename(s)
    s = re.sub(r'\.(wav|mp3|flac)$', '', s)
    s = s.replace('_aug', '')
    return s


def merge_clean_and_augmented(clean_csv, augmented_csv, out_csv):
    print("=== Merge: Clean + Augmented Datasets (robust) ===")

    df_clean = pd.read_csv(clean_csv, low_memory=False)
    df_aug = pd.read_csv(augmented_csv, low_memory=False)

    # Einheitliche Spaltennamen
    if "file" in df_clean.columns:
        df_clean.rename(columns={"file": "filename"}, inplace=True)
    if "file" in df_aug.columns:
        df_aug.rename(columns={"file": "filename"}, inplace=True)

    df_clean["source"] = "clean"
    df_aug["source"] = "augmented"

    # Normalisierte Spalte zum Gruppieren
    df_clean["group_id"] = df_clean["filename"].map(normalize_filename)
    df_aug["group_id"] = df_aug["filename"].map(normalize_filename)

    print(f"Clean: {len(df_clean)}  |  Augmented: {len(df_aug)}")

    # Warnung bei Spaltenabweichungen
    missing_cols = set(df_clean.columns) ^ set(df_aug.columns)
    if missing_cols:
        print(f"Achtung: Unterschiedliche Spalten erkannt: {missing_cols}")

    # Vereinigung
    df_all = pd.concat([df_clean, df_aug], axis=0, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_all.to_csv(out_csv, index=False)

    print(f"✅ Kombinierter Datensatz gespeichert unter: {out_csv}")
    print(f"Gesamt: {len(df_all)} Zeilen, {len(df_all.columns)} Spalten")
    print(f"Spalten: {', '.join(df_all.columns[:10])} ...")


def main():
    parser = argparse.ArgumentParser(description="Merge clean + augmented datasets with robust matching")
    parser.add_argument("--clean_csv", required=True, help="Pfad zum cleanen Datensatz (z. B. merged_sigmos_wavlm_multiwer.csv)")
    parser.add_argument("--augmented_csv", required=True, help="Pfad zum augmentierten Datensatz (z. B. merged_sigmos_wavlm_augmented_esc50_snr15_30.csv)")
    parser.add_argument("--out_csv", required=True, help="Pfad zur kombinierten Ausgabedatei")
    args = parser.parse_args()

    merge_clean_and_augmented(args.clean_csv, args.augmented_csv, args.out_csv)


if __name__ == "__main__":
    main()