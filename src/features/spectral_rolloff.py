import librosa
import numpy as np

def calculate_spectral_rolloff(audio_signal, sample_rate=16000, roll_percent=0.85):
    """
    Berechnet den Spectral Rolloff eines Audiosignals.

    Quelle:
        - Librosa: spectral_rolloff
        - https://librosa.org/doc/main/generated/librosa.feature.spectral_rolloff.html

    Args:
        audio_signal (np.array): Vorverarbeitetes Audio-Signal.
        sample_rate (int): Sampling-Rate des Audiosignals.
        roll_percent (float): Anteil der Energie, unterhalb dessen die Grenzfrequenz berechnet wird (Standard: 85%).

    Returns:
        float: Durchschnittlicher Spectral Rolloff-Wert über alle Frames (Hz).
    """
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sample_rate, roll_percent=roll_percent)

    # Mittelwert über alle Frames berechnen
    return np.mean(spectral_rolloff)