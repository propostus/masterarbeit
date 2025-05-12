import librosa
import numpy as np

def calculate_chroma_features(audio_signal, sample_rate=16000, n_fft=1024, hop_length=512):
    """
    Berechnet die mittleren Chroma-Features eines Audiosignals.

    Quelle:
        - Librosa: Chroma Features (Tonhöhenverteilung)
        - https://librosa.org/doc/main/generated/librosa.feature.chroma_stft.html

    Args:
        audio_signal (np.array): Vorverarbeitetes Audio-Signal.
        sample_rate (int): Sampling-Rate des Audiosignals.
        n_fft (int): Anzahl der FFT-Punkte für die Spektralanalyse.
        hop_length (int): Schrittweite zwischen FFT-Berechnungen.

    Returns:
        np.array: Mittlere Chroma-Werte für jede der 12 Tonhöhenklassen.
    """
    # Berechnung der Chroma Features
    chroma = librosa.feature.chroma_stft(y=audio_signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)

    # Mittelwert über alle Frames berechnen
    return np.mean(chroma, axis=1)