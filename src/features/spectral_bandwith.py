import numpy as np
import librosa

def calculate_spectral_bandwidth(audio_signal, sample_rate=16000, frame_length=2048, hop_length=512):
    """
    Berechnet die Spectral Bandwidth eines Audiosignals.

    Quelle:
        Peeters, G. (2004). A large set of audio features for sound description (similarity and classification).
        Technical Report, IRCAM.

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).
        sample_rate (int): Sampling-Rate des Signals (Standard: 16000 Hz).
        frame_length (int): Länge eines Frames in Samples (Standard: 2048).
        hop_length (int): Schrittweite zwischen Frames in Samples (Standard: 512).

    Returns:
        float: Durchschnittliche Spectral Bandwidth in Hz.
    """
    spectral_bw = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sample_rate, n_fft=frame_length, hop_length=hop_length)

    # Mittelwert über alle Frames berechnen
    return np.mean(spectral_bw)