import librosa
import numpy as np

def preprocess_audio(audio_path, sample_rate=16000):
    """
    Vorverarbeitung einer Audiodatei: Resampling und Normalisierung.

    Args:
        audio_path (str): Pfad zur Audiodatei.
        sample_rate (int): Ziel-Sampling-Rate (Standard: 16000).

    Returns:
        np.array: Das vorverarbeitete Audio-Signal.
    """
    # Audiodatei laden und resamplen
    audio_signal, _ = librosa.load(audio_path, sr=sample_rate)

    # Zu Mono konvertieren (falls mehrkanalig)
    audio_signal = librosa.to_mono(audio_signal)

    # Normalisieren
    max_val = np.max(np.abs(audio_signal))
    if max_val > 0:
        audio_signal = audio_signal / max_val

    return audio_signal