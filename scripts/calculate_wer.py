# scripts/calculate_wer.py
import os
import argparse
import pandas as pd
import whisper
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip, RemoveMultipleSpaces

# === Standard Text-Normalisierung ===
STANDARD_TRANSFORM = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip(),
    RemoveMultipleSpaces(),
])

def normalize_text(s: str) -> str:
    """Normalisiert Transkripttexte (Kleinbuchstaben, keine Satzzeichen, etc.)."""
    return STANDARD_TRANSFORM(s if isinstance(s, str) else str(s))

def normalize_name(s: str) -> str:
    """Normalisiert Dateinamen (nur Basename, lowercase)."""
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
    """Transkribiert Audiodateien mit Whisper und berechnet WER."""
    model = whisper.load_model(model_name)

    # TSV-Datei laden
    try:
        df = pd.read_csv(tsv_path, sep="\t", usecols=["path", "sentence"])
    except ValueError:
        raise RuntimeError("Die TSV-Datei muss Spalten 'path' und 'sentence' enthalten.")

    # Optional: nur bestimmte Dateien verarbeiten
    restrict_set = None
    if restrict_to_features_csv:
        df_feat = pd.read_csv(restrict_to_features_csv, usecols=["filename"])
        restrict_set = set(df_feat["filename"].map(normalize_name))

    results = []
    processed = 0
    tried = 0

    for _, row in df.iterrows():
        filename_mp3 = row["path"]
        reference_raw = str(row["sentence"]).strip()

        # MP3 zu WAV umwandeln (da noisy-Dateien .wav heißen)
        filename = os.path.splitext(filename_mp3)[0] + ".wav"
        file_path = os.path.join(audio_dir, filename)

        tried += 1
        if not os.path.exists(file_path):
            continue  # Datei nicht gefunden, überspringen

        # Optional: nur bestimmte Dateien verarbeiten
        if restrict_set is not None and normalize_name(filename) not in restrict_set:
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

    # Ergebnisse speichern
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)

    print(f"WER-Ergebnisse gespeichert unter: {out_csv}")
    print(f"Dateien geprüft: {tried}, erfolgreich transkribiert: {processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Berechne WER mit Whisper (für clean oder noisy Audiodateien)")
    parser.add_argument("--audio_dir", required=True, help="Ordner mit Audiodateien (z. B. .../snr_20)")
    parser.add_argument("--tsv_path", required=True, help="Pfad zu Common Voice validated.tsv")
    parser.add_argument("--out_csv", required=True, help="Pfad zur Ergebnis-CSV")
    parser.add_argument("--model", default="tiny", help="Whisper-Modell: tiny|base|small|medium|large")
    parser.add_argument("--language", default="de", help="Sprache, z. B. de, en, fr")
    parser.add_argument("--max_files", type=int, default=None, help="Maximale Anzahl Dateien (Debug/Test)")
    parser.add_argument("--restrict_to_features_csv", type=str, default=None,
                        help="Optional: Nur Dateien berücksichtigen, die in dieser Feature-CSV stehen (Spalte 'filename').")
    args = parser.parse_args()

    transcribe_and_calculate_wer(
        audio_dir=args.audio_dir,
        tsv_path=args.tsv_path,
        out_csv=args.out_csv,
        model_name=args.model,
        language=args.language,
        max_files=args.max_files,
        restrict_to_features_csv=args.restrict_to_features_csv,
    )