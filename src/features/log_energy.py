import numpy as np

def calculate_log_energy(audio_signal):
    """
    Berechnet die logarithmische Energie eines Audiosignals.

    Quelle:
        ETSI ES 201 108 V1.1.2 (2000-04), Speech Processing, Transmission and Quality aspects (STQ);
        Distributed speech recognition; Front-end feature extraction algorithm; Compression algorithms.

    Args:
        audio_signal (np.array): Das Audio-Signal (1D-Array).

    Returns:
        float: Die logarithmische Energie.
    """
    # Quadrieren der Amplituden und Summieren
    energy = np.sum(audio_signal**2)

    # Logarithmus berechnen mit Floor-Wert aus dem ETSI-Standard
    log_energy = np.log(energy + 2e-22)  # Offset: 2e-22, um log(0) zu vermeiden
    return log_energy