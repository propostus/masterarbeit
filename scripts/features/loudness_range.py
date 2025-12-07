# src/features/loudness_range.py
import numpy as np

def compute(signal: np.ndarray, sr: int,
            lower_percentile: int = 10,
            upper_percentile: int = 95) -> dict:
    """
    Berechnet die Loudness Range (LRA) eines Audiosignals als Perzentil-Bereich.
    (vereinfachte Version nach EBU TECH 3342)
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz (hier nicht genutzt, nur für API-Konsistenz).
        lower_percentile (int): Unteres Perzentil (Default: 10).
        upper_percentile (int): Oberes Perzentil (Default: 95).
    
    Returns:
        dict: {"lra_db": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"lra_db": np.nan}

    abs_signal = np.abs(signal)

    p_upper = np.percentile(abs_signal, upper_percentile)
    p_lower = np.percentile(abs_signal, lower_percentile)

    if p_lower <= 1e-12:  # praktisch Stille
        return {"lra_db": np.nan}

    lra = 20 * np.log10(p_upper / p_lower)
    return {"lra_db": float(lra)}