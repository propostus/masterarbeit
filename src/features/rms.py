import numpy as np

def calculate_rms(audio_signal):
    """
    Berechnet den Root Mean Square (RMS) eines Audiosignals.

    Quelle:
        Andreas Friesecke, Die Audio-Enzyklopädie, K. G. Saur, 2007, S. 208,
        ISBN: 978-3-598-11774-9
    
    Args:
        audio_signal (np.array): Das Audio-Signal (1D-Array).

    Returns:
        float: Der RMS-Wert.
    """
    return np.sqrt(np.mean(audio_signal**2))