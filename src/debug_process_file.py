from src.compute_features_parallel_tqdm import process_file
import numpy as np
np.seterr(all='warn')
# Beispielhafte MP3-Datei aus deinem Datensatz
file_path = "audio_files/common_voice/raw/cv-corpus-de-combined-20-21-delta/common_voice_de_41249729.mp3"

# Aufruf testen
result = process_file(file_path)
print(result)