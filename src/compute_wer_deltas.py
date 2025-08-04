import os
import pandas as pd
import whisper
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces
from tqdm import tqdm

# Whisper-Modell laden
model_size = "tiny"
device = "cpu"
model = whisper.load_model(model_size, device=device)

# Normalisierung definieren
transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces()
])

# Konfigurationen für die 4 Delta-Segmente
datasets = [
    {
        "name": "cv-de-20",
        "tsv_path": "audio_files/common_voice/raw/cv-corpus-de-22.0-2025-06-20/de/validated.tsv",
        "audio_dir": "audio_files/common_voice/raw/cv-corpus-de-20.0-delta-2024-12-06/de/clips",
        "language": "de"
    },
    {
        "name": "cv-de-21",
        "tsv_path": "audio_files/common_voice/raw/cv-corpus-de-22.0-2025-06-20/de/validated.tsv",
        "audio_dir": "audio_files/common_voice/raw/cv-corpus-de-21.0-delta-2025-03-14/de/clips",
        "language": "de"
    },
    {
        "name": "cv-en-20",
        "tsv_path": "audio_files/common_voice/raw/cv-corpus-en-22.0-2025-06-20/en/validated.tsv",
        "audio_dir": "audio_files/common_voice/raw/cv-corpus-en-20.0-delta-2024-12-06/en/clips",
        "language": "en"
    },
    {
        "name": "cv-en-21",
        "tsv_path": "audio_files/common_voice/raw/cv-corpus-en-22.0-2025-06-20/en/validated.tsv",
        "audio_dir": "audio_files/common_voice/raw/cv-corpus-en-21.0-delta-2025-03-14/en/clips",
        "language": "en"
    },
]

# Ergebnisse-Verzeichnis
results_dir = "results/wer_deltas"
os.makedirs(results_dir, exist_ok=True)

for data in datasets:
    print(f"\n🔍 Bearbeite: {data['name']}")
    
    try:
        df = pd.read_csv(data["tsv_path"], sep="\t")
        files = os.listdir(data["audio_dir"])
        df = df[df["path"].isin(files)]

        if df.empty:
            print(f"⚠️ Keine Übereinstimmungen in {data['name']}, überspringe.")
            continue

        results = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            filename = row["path"]
            reference = str(row["sentence"]).strip()
            file_path = os.path.join(data["audio_dir"], filename)

            if not os.path.exists(file_path):
                continue

            try:
                result = model.transcribe(file_path, language=data["language"], fp16=False)
                hypothesis = result["text"].strip()
                wer_score = wer(transform(reference), transform(hypothesis))

                results.append({
                    "filename": filename,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": wer_score
                })

            except Exception as e:
                print(f"Fehler bei {filename}: {e}")
                continue

        # Speichern
        result_df = pd.DataFrame(results)
        output_path = os.path.join(results_dir, f"{data['name']}_wer.csv")
        result_df.to_csv(output_path, index=False)
        print(f"✅ Ergebnisse gespeichert: {output_path} ({len(result_df)} Dateien)")

    except Exception as e:
        print(f"❌ Fehler bei {data['name']}: {e}")