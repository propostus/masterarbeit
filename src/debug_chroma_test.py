import librosa
from src.preprocessing import preprocess_audio
from src.features.chroma_features import calculate_chroma_features

file_path = "audio_files/common_voice/raw/cv-corpus-de-combined-20-21-delta/common_voice_de_41249729.mp3"
sample_rate = 16000

# Audio laden
audio = preprocess_audio(file_path, sample_rate=sample_rate)

# Chroma berechnen
chroma = calculate_chroma_features(audio, sample_rate=sample_rate)

print("Chroma:", chroma)