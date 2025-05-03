import numpy as np

def calculate_crest_factor(audio_signal):
    """
    Berechnet den Crest Factor eines Audiosignals.

    Quelle:
        Peeters, G. (2004). A large set of audio features for sound description (similarity and classification).
        Technical Report, IRCAM.

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).

    Returns:
        float: Crest Factor des Signals.
    """
    # Spitzenamplitude (Maximalwert der absoluten Amplitude)
    peak_amplitude = np.max(np.abs(audio_signal))
    
    # RMS-Wert berechnen
    rms_value = np.sqrt(np.mean(audio_signal ** 2))

    # Crest Factor berechnen
    if rms_value > 0:
        crest_factor = peak_amplitude / rms_value
    else:
        crest_factor = 0.0  # Falls RMS = 0, setzen wir den Crest Factor auf 0

    return crest_factor