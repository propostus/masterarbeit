import os
import torch
import pandas as pd
import whisper
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces
from tqdm import tqdm

# Parameter
model_size = "large"  # z. B. tiny, base, small, medium, large
device = "cpu"
tsv_path = "audio_files/cv-corpus-21.0-2025-03-14/en/validated.tsv"
audio_dir = "audio_files/common_voice_subset_10h/"
output_path = "results/subset_10h/wer_test_10_large.csv"

# Whisper-Modell laden
model = whisper.load_model(model_size, device=device)

# TSV einlesen
df = pd.read_csv(tsv_path, sep="\t")
df = df[df["path"].isin(os.listdir(audio_dir))]

# Normalisierung definieren
transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces()
])

results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    filename = row["path"]
    reference = str(row["sentence"]).strip()

    file_path = os.path.join(audio_dir, filename)
    if not os.path.exists(file_path):
        continue

    try:
        result = model.transcribe(file_path, language="en", fp16=False)
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

# Ergebnis speichern
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"\n✅ WER-Ergebnisse gespeichert in: {output_path}")