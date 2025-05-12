import librosa
import numpy as np

def calculate_spectral_contrast(audio_signal, sample_rate=16000, n_bands=6):
    """
    Berechnet den Spectral Contrast eines Audiosignals.

    Quelle:
        - Librosa: spectral_contrast
        - https://librosa.org/doc/main/generated/librosa.feature.spectral_contrast.html

    Args:
        audio_signal (np.array): Vorverarbeitetes Audio-Signal.
        sample_rate (int): Sampling-Rate des Audiosignals.
        n_bands (int): Anzahl der Frequenzbänder für die Kontrastanalyse.

    Returns:
        np.array: Mittelwerte der Spektral-Kontraste über die Bänder.
    """
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_signal, sr=sample_rate, n_bands=n_bands)

    # Mittelwert über alle Frames berechnen
    return np.mean(spectral_contrast, axis=1)