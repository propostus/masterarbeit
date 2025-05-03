import numpy as np
import librosa

def calculate_f0(audio_signal, sample_rate=16000, frame_length=2048, hop_length=512, fmin=75, fmax=300):
    """
    Berechnet die Fundamental Frequency (F0) eines Audiosignals mit der pYIN-Methode.

    Quelle:
        Mauch, M., & Dixon, S. (2014). pYIN: A fundamental frequency estimator using probabilistic threshold distributions.
        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).
        sample_rate (int): Sampling-Rate des Signals (Standard: 16000 Hz).
        frame_length (int): Länge eines Frames in Samples (Standard: 2048).
        hop_length (int): Schrittweite zwischen Frames in Samples (Standard: 512).
        fmin (int): Minimale erwartete Frequenz in Hz (Standard: 75 Hz für Sprache).
        fmax (int): Maximale erwartete Frequenz in Hz (Standard: 300 Hz für Sprache).

    Returns:
        float: Durchschnittliche Fundamental Frequency (F0) in Hz.
    """
    # Berechnung der Grundfrequenz mit pYIN (eine verbesserte Version von YIN)
    f0_values, voiced_flag, _ = librosa.pyin(
        audio_signal, fmin=fmin, fmax=fmax, sr=sample_rate, frame_length=frame_length, hop_length=hop_length
    )

    # Nur die Werte berücksichtigen, die als "voiced" erkannt wurden (keine NaN-Werte)
    f0_values = f0_values[~np.isnan(f0_values)]

    # Durchschnittliche Grundfrequenz berechnen
    average_f0 = np.mean(f0_values) if len(f0_values) > 0 else 0.0
    return average_f0