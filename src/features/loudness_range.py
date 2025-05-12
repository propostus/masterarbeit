import numpy as np

def calculate_loudness_range(audio_signal, lower_percentile=10, upper_percentile=95):
    """
    Berechnet die Loudness Range (LRA) eines Audiosignals basierend auf EBU TECH 3342.

    Quelle:
        EBU TECH 3342: "LOUDNESS RANGE: A MEASURE TO SUPPLEMENT EBU R 128 LOUDNESS NORMALIZATION"
        Geneva, November 2023.

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).
        lower_percentile (int): Unteres Perzentil (Standard: 10).
        upper_percentile (int): Oberes Perzentil (Standard: 95).

    Returns:
        float: Loudness Range (LRA) in Dezibel.
    """
    # Absolutwerte des Signals für die Perzentilberechnung
    abs_signal = np.abs(audio_signal)
    
    # Perzentile berechnen
    p_upper = np.percentile(abs_signal, upper_percentile)
    p_lower = np.percentile(abs_signal, lower_percentile)
    
    # LRA berechnen
    if p_lower > 0:  # Verhindert Division durch Null
        lra = 20 * np.log10(p_upper / p_lower)
    else:
        lra = 0.0  # Falls p_lower = 0, setzen wir die LRA auf 0 dB
    
    return lra