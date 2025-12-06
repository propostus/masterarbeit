# scripts/merge_cv23_features.py
# ------------------------------------------------------
# Merged:
#   - WavLM Embeddings (mean+std)
#   - SigMOS MOS-Features
#   - WER tiny/base/small
#   - Common Voice 23 Meta (client_id, age, gender, sentence)
#
# Annahmen:
#   - Alle Quellen beziehen sich auf dieselben Clips (balanced, <=10s, max 50 pro Sprecher)
#   - Dateinamen können in Input-CSVs "file", "filename" oder "path" heißen
#   - validated_balanced_final.tsv enthält Spalten: path, client_id, age, gender, sentence, ...
# ------------------------------------------------------

import os
import argparse
import pandas as pd
import numpy as np


def find_name_column(df, candidates=("filename", "file", "path")):
    """Finde die Spalte, die den Dateinamen enthält."""
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"Keine Name-Spalte gefunden. Erwartet eine von {candidates}, gefunden: {list(df.columns)}")


def normalize_filename(series):
    """Auf Basename + lowercase normalisieren."""
    return series.astype(str).str.strip().apply(os.path.basename).str.lower()


def load_wavlm(path):
    print(f"Lade WavLM-Embeddings von: {path}")
    df = pd.read_csv(path, low_memory=False)
    name_col = find_name_column(df)
    df = df.copy()
    df.rename(columns={name_col: "filename"}, inplace=True)
    df["filename"] = normalize_filename(df["filename"])
    print(f"WavLM: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    return df


def load_sigmos(path):
    print(f"Lade SigMOS-Features von: {path}")
    df = pd.read_csv(path, low_memory=False)
    name_col = find_name_column(df)
    df = df.copy()
    df.rename(columns={name_col: "filename"}, inplace=True)
    df["filename"] = normalize_filename(df["filename"])
    print(f"SigMOS: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    return df


def load_wer(path, target_name):
    """
    Erwartet eine CSV mit:
      - filename oder file
      - wer
      - optional reference, hypothesis
    """
    print(f"Lade WER ({target_name}) von: {path}")
    df = pd.read_csv(path, low_memory=False)
    name_col = find_name_column(df)
    if "wer" not in df.columns:
        raise RuntimeError(f"In {path} wurde keine Spalte 'wer' gefunden.")
    df = df.copy()
    df.rename(columns={name_col: "filename", "wer": target_name}, inplace=True)
    df["filename"] = normalize_filename(df["filename"])
    cols_keep = ["filename", target_name]
    # Referenz/Hypothese behalten wir hier bewusst nicht; sie sind für Training nicht nötig
    df = df[cols_keep]
    print(f"WER {target_name}: {df.shape[0]} Zeilen")
    return df


def load_meta_tsv(path):
    """
    Lädt validated_balanced_final.tsv und extrahiert Meta-Information:
      - filename (aus path)
      - client_id, age, gender, sentence
    """
    print(f"Lade Meta-Tabelle von: {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "path" not in df.columns:
        raise RuntimeError(f"{path} enthält keine Spalte 'path'.")
    df = df.copy()
    df["filename"] = normalize_filename(df["path"])

    cols_meta = ["filename"]
    for c in ["client_id", "age", "gender", "sentence"]:
        if c in df.columns:
            cols_meta.append(c)

    meta = df[cols_meta].drop_duplicates(subset=["filename"])
    print(f"Meta: {meta.shape[0]} eindeutige Dateien, Spalten: {cols_meta}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Merge CV23 balanced Features (WavLM + SigMOS + WER + Meta)")
    parser.add_argument("--wavlm_csv", required=True, help="Pfad zur WavLM-Embeddings CSV")
    parser.add_argument("--sigmos_csv", required=True, help="Pfad zur SigMOS-Features CSV")
    parser.add_argument("--wer_tiny_csv", required=True, help="Pfad zur WER CSV (tiny)")
    parser.add_argument("--wer_base_csv", required=True, help="Pfad zur WER CSV (base)")
    parser.add_argument("--wer_small_csv", required=True, help="Pfad zur WER CSV (small)")
    parser.add_argument("--validated_tsv", required=True, help="Pfad zu validated_balanced_final.tsv")
    parser.add_argument("--out_csv", required=True, help="Pfad zur Ausgabe-CSV")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # 1) Basis-Features laden
    wavlm = load_wavlm(args.wavlm_csv)
    sigmos = load_sigmos(args.sigmos_csv)

    # Inner Join: wir behalten nur Dateien, für die beide Embedding-Typen vorliegen
    print("\nMerging WavLM + SigMOS (inner join auf filename) ...")
    df = pd.merge(wavlm, sigmos, on="filename", how="inner")
    print(f"Nach Merge WavLM+SigMOS: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # 2) WER-Daten laden und mergen
    wer_tiny = load_wer(args.wer_tiny_csv, "wer_tiny")
    wer_base = load_wer(args.wer_base_csv, "wer_base")
    wer_small = load_wer(args.wer_small_csv, "wer_small")

    print("\nMerging WER tiny/base/small ...")
    df = df.merge(wer_tiny, on="filename", how="left")
    df = df.merge(wer_base, on="filename", how="left")
    df = df.merge(wer_small, on="filename", how="left")
    print(f"Nach Merge mit WER: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # 3) Meta aus validated_balanced_final.tsv hinzufügen
    meta = load_meta_tsv(args.validated_tsv)
    print("\nMerging Meta-Information (client_id, age, gender, sentence) ...")
    df = df.merge(meta, on="filename", how="left")
    print(f"Nach Merge mit Meta: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

    # 4) Doppelte Dateinamen prüfen
    dup_count = df["filename"].duplicated().sum()
    if dup_count > 0:
        print(f"Achtung: {dup_count} doppelte filename-Einträge nach dem Merge.")
    else:
        print("Keine doppelten filename-Einträge nach dem Merge.")

    # 5) Kurze Übersicht
    n_total = df.shape[0]
    n_with_all_wer = df.dropna(subset=["wer_tiny", "wer_base", "wer_small"]).shape[0]
    print(f"\nGesamtzeilen: {n_total}")
    print(f"Zeilen mit allen drei WERs: {n_with_all_wer}")
    print(f"Anteil mit vollständigen Targets: {n_with_all_wer / n_total:.3f}")

    # 6) Speichern
    df.to_csv(args.out_csv, index=False)
    print(f"\nFertig. Merged Dataset gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()