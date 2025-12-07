# src/features/crest_factor.py
import numpy as np

def compute(signal: np.ndarray, sr: int) -> dict:
    """
    Berechnet den Crest Factor eines Audiosignals (Peak/RMS).
    
    Quelle:
        Peeters, G. (2004). 
        A large set of audio features for sound description (similarity and classification).
        Technical Report, IRCAM.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz (hier nicht genutzt, nur für API-Konsistenz).
    
    Returns:
        dict: {"crest_factor": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"crest_factor": np.nan}

    peak_amplitude = np.max(np.abs(signal))
    rms_value = np.sqrt(np.mean(signal**2))

    if rms_value <= 1e-12:
        return {"crest_factor": np.nan}

    crest_factor = peak_amplitude / rms_value
    return {"crest_factor": float(crest_factor)}