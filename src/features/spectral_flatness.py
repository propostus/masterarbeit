import numpy as np
import librosa

def calculate_spectral_flatness(audio_signal, frame_length=2048, hop_length=512):
    """
    Berechnet die Spectral Flatness eines Audiosignals.

    Quelle:
        Peeters, G. (2004). A large set of audio features for sound description (similarity and classification).
        Technical Report, IRCAM.

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).
        frame_length (int): Länge eines Frames in Samples (Standard: 2048).
        hop_length (int): Schrittweite zwischen Frames in Samples (Standard: 512).

    Returns:
        float: Durchschnittlicher Spectral Flatness-Wert (zwischen 0 und 1).
    """
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_signal, n_fft=frame_length, hop_length=hop_length)

    # Mittelwert über alle Frames berechnen
    return np.mean(spectral_flatness)