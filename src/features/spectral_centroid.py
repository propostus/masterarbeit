import numpy as np
import librosa

def calculate_spectral_centroid(audio_signal, sample_rate=16000, frame_length=2048, hop_length=512):
    """
    Berechnet den Spectral Centroid eines Audiosignals.

    Quelle:
        Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals.
        IEEE Transactions on Speech and Audio Processing.

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).
        sample_rate (int): Sampling-Rate des Signals (Standard: 16000 Hz).
        frame_length (int): Länge eines Frames in Samples (Standard: 2048).
        hop_length (int): Schrittweite zwischen Frames in Samples (Standard: 512).

    Returns:
        float: Durchschnittlicher Spectral Centroid in Hz.
    """
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_signal, sr=sample_rate, n_fft=frame_length, hop_length=hop_length)
    
    # Mittelwert über alle Frames berechnen
    return np.mean(spectral_centroid)