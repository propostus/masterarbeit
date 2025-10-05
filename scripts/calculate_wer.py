# scripts/calculate_wer.py
import os
import argparse
import pandas as pd
import whisper
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip, RemoveMultipleSpaces

STANDARD_TRANSFORM = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip(),
    RemoveMultipleSpaces(),
])

def normalize_text(s: str) -> str:
    return STANDARD_TRANSFORM(s if isinstance(s, str) else str(s))

def normalize_name(s: str) -> str:
    s = str(s).strip()
    s = os.path.basename(s)
    return s.lower()

def transcribe_and_calculate_wer(
    audio_dir: str,
    tsv_path: str,
    out_csv: str,
    model_name: str = "tiny",
    language: str = "de",
    max_files: int | None = None,
    restrict_to_features_csv: str | None = None,
):
    model = whisper.load_model(model_name)

    # TSV minimal laden
    try:
        df = pd.read_csv(tsv_path, sep="\t", usecols=["path", "sentence"])
    except ValueError:
        raise RuntimeError("Die TSV-Datei muss Spalten 'path' und 'sentence' enthalten.")

    # Optional: auf Feature-Dateiliste einschränken
    restrict_set = None
    if restrict_to_features_csv:
        df_feat = pd.read_csv(restrict_to_features_csv, usecols=["filename"])
        restrict_set = set(df_feat["filename"].map(normalize_name))

    results = []
    processed = 0

    for _, row in df.iterrows():
        filename = row["path"]
        reference_raw = str(row["sentence"]).strip()
        candidate_key = normalize_name(filename)

        # Falls eingeschränkt: nur verarbeiten, wenn in der Feature-Liste
        if restrict_set is not None and candidate_key not in restrict_set:
            continue

        file_path = os.path.join(audio_dir, filename)
        if not os.path.exists(file_path):
            continue

        if max_files is not None and processed >= max_files:
            break

        try:
            result = model.transcribe(file_path, language=language, fp16=False)
            hypothesis_raw = result["text"].strip()

            reference = normalize_text(reference_raw)
            hypothesis = normalize_text(hypothesis_raw)

            score = wer(reference, hypothesis)

            results.append({
                "filename": filename,
                "reference": reference_raw,
                "hypothesis": hypothesis_raw,
                "wer": score
            })
            processed += 1
            print(f"[{processed}] {filename}: WER={score:.3f}")

        except Exception as e:
            print(f"Fehler bei {filename}: {e}")

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"WER-Ergebnisse gespeichert unter: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Ordner mit Audiodateien (z. B. .../clips)")
    parser.add_argument("--tsv_path", required=True, help="Pfad zu Common Voice validated.tsv")
    parser.add_argument("--out_csv", required=True, help="Pfad zur Ergebnis-CSV")
    parser.add_argument("--model", default="tiny", help="Whisper-Modell: tiny|base|small|medium|large")
    parser.add_argument("--language", default="de", help="Sprache, z. B. de, en, fr")
    parser.add_argument("--max_files", type=int, default=None, help="Maximale Anzahl Dateien")
    parser.add_argument("--restrict_to_features_csv", type=str, default=None,
                        help="Optional: Nur Dateien berücksichtigen, die in dieser Feature-CSV stehen (Spalte 'filename').")
    args = parser.parse_args()

    transcribe_and_calculate_wer(
        args.audio_dir,
        args.tsv_path,
        args.out_csv,
        model_name=args.model,
        language=args.language,
        max_files=args.max_files,
        restrict_to_features_csv=args.restrict_to_features_csv,
    )