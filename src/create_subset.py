import os
import random
import shutil
import librosa

# === PARAMETER ===
source_dir = "audio_files/cv-corpus-21.0-2025-03-14/en/clips"           # Ordner mit allen MP3-Dateien
target_dir = "audio_files/common_voice_subset_10h/"                     # Zielordner für das Subset
sample_rate = 16000
target_duration_hours = 10
audio_extension = ".mp3"

# === VORBEREITUNG ===
os.makedirs(target_dir, exist_ok=True)
all_files = []

# Alle MP3-Dateien rekursiv sammeln
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(audio_extension):
            full_path = os.path.join(root, file)
            all_files.append(full_path)

print(f"📂 {len(all_files)} Audiodateien gefunden.")

# === Zufällig mischen ===
random.seed(42)
random.shuffle(all_files)

# === Subset iterativ aufbauen ===
selected_files = []
total_duration_sec = 0
max_duration_sec = target_duration_hours * 3600

print("⏳ Wähle Dateien bis ca. 10 Stunden...")

for path in all_files:
    try:
        duration = librosa.get_duration(path=path, sr=sample_rate)
        if total_duration_sec + duration > max_duration_sec:
            break
        total_duration_sec += duration
        selected_files.append(path)
    except Exception as e:
        print(f"❌ Fehler bei Datei {path}: {e}")
        continue

print(f"✅ {len(selected_files)} Dateien gewählt ({round(total_duration_sec / 3600, 2)} h Gesamtzeit).")

# === Kopieren in Zielordner ===
for src_path in selected_files:
    filename = os.path.basename(src_path)
    target_path = os.path.join(target_dir, filename)
    shutil.copy2(src_path, target_path)

print(f"📁 Dateien kopiert nach: {target_dir}")