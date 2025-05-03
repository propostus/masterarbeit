import numpy as np

def calculate_clipping_ratio(audio_signal, threshold=0.95):
    """
    Berechnet die Clipping Ratio eines Audiosignals.

    Quelle:
        ???

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array, Werte zwischen -1 und 1).
        threshold (float): Clipping-Schwellenwert (Standard: 0.99).

    Returns:
        float: Clipping Ratio (Anteil der geclippten Samples am gesamten Signal).
    """
    # Anzahl der geclippten Samples bestimmen
    clipped_samples = np.sum(np.abs(audio_signal) >= threshold)

    # Clipping Ratio berechnen
    clipping_ratio = clipped_samples / len(audio_signal)
    
    return clipping_ratio