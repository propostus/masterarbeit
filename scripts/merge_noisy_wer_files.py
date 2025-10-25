# scripts/merge_noisy_wer_files.py
import os
import argparse
import pandas as pd

def merge_noisy_wer_files(wer_dir: str, out_dir: str):
    """
    Erwartete Eingaben im wer_dir:
      wer_tiny_snr0.csv, wer_tiny_snr10.csv, wer_tiny_snr20.csv
      wer_small_snr0.csv, ...
      wer_base_snr0.csv, ...

    Ausgabe im out_dir:
      wer_tiny_noisy.csv, wer_small_noisy.csv, wer_base_noisy.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    models = ["tiny", "small", "base"]
    snr_levels = [0, 10, 20]

    for model in models:
        parts = []
        for snr in snr_levels:
            fp = os.path.join(wer_dir, f"wer_{model}_snr{snr}.csv")
            if not os.path.exists(fp):
                print(f"Warnung: {fp} nicht gefunden – übersprungen")
                continue

            df = pd.read_csv(fp)
            # Vereinheitliche Spaltennamen defensiv
            cols = {c.lower().strip(): c for c in df.columns}
            # sichere Umbenennung falls nötig
            rename_map = {}
            for std in ["filename", "reference", "hypothesis", "wer"]:
                if std not in df.columns and std in cols:
                    rename_map[cols[std]] = std
            if rename_map:
                df = df.rename(columns=rename_map)

            # Pflichtspalten prüfen
            missing = [c for c in ["filename", "wer"] if c not in df.columns]
            if missing:
                print(f"Übersprungen (fehlende Spalten {missing}): {fp}")
                continue

            df["snr"] = snr
            # nur relevante Spalten behalten + stabile Reihenfolge
            keep = [c for c in ["filename", "reference", "hypothesis", "wer", "snr"] if c in df.columns]
            parts.append(df[keep])

            print(f"Lade {fp} ({len(df)} Zeilen)")

        if not parts:
            print(f"Keine noisy-WER-Dateien für {model} gefunden.")
            continue

        merged = pd.concat(parts, ignore_index=True)
        # sortiert für Reproduzierbarkeit
        merged = merged.sort_values(by=["filename", "snr"]).reset_index(drop=True)

        out_path = os.path.join(out_dir, f"wer_{model}_noisy.csv")
        merged.to_csv(out_path, index=False)
        print(f"Gespeichert: {out_path} ({len(merged)} Zeilen)")

def main():
    parser = argparse.ArgumentParser(
        description="Merge WER-Dateien der noisy-Datensätze pro Modell (tiny/base/small) zu je einer CSV."
    )
    parser.add_argument("--wer_dir", required=True,
                        help="Verzeichnis mit noisy-WER-CSV-Dateien (z. B. results/wer/noisy)")
    parser.add_argument("--out_dir", required=True,
                        help="Zielverzeichnis für die gemergten Dateien (z. B. results/wer/noisy_merged)")
    args = parser.parse_args()

    merge_noisy_wer_files(args.wer_dir, args.out_dir)

if __name__ == "__main__":
    main()