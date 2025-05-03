import numpy as np
import librosa

def calculate_phoneme_entropy(audio_signal, sample_rate=16000, frame_length=2048, hop_length=512):
    """
    Berechnet die Phoneme Entropy eines Audiosignals basierend auf der spektralen Energieverteilung.

    Quelle:
        Jelinek, F. (1997). Statistical Methods for Speech Recognition.
        MIT Press.

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).
        sample_rate (int): Sampling-Rate des Signals (Standard: 16000 Hz).
        frame_length (int): Länge eines FFT-Frames in Samples (Standard: 2048).
        hop_length (int): Schrittweite zwischen Frames in Samples (Standard: 512).

    Returns:
        float: Phoneme Entropy-Wert für das gesamte Signal.
    """
    # Falls das Signal zu leise oder konstant ist, Entropie auf 0 setzen
    if np.all(audio_signal == audio_signal[0]):
        return 0.0

    # Berechnung des Leistungsverhältnisses (Power Spectrogram)
    spectrogram = np.abs(librosa.stft(audio_signal, n_fft=frame_length, hop_length=hop_length)) ** 2

    # Normalisierung über die Frequenzbänder (damit die Summe 1 ergibt)
    prob_distribution = spectrogram / np.sum(spectrogram, axis=0, keepdims=True)

    # Berechnung der Entropie pro Frame (log2, um Bits als Einheit zu haben)
    entropy_per_frame = -np.sum(prob_distribution * np.log2(prob_distribution + 1e-10), axis=0)

    # Mittelwert über alle Frames
    entropy_value = np.mean(entropy_per_frame)

    return entropy_value