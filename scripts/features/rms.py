# src/features/rms.py
import numpy as np

def compute(signal: np.ndarray, sr: int) -> dict:
    """
    Berechnet den RMS-Wert (Root Mean Square) eines Audiosignals.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
    
    Returns:
        dict: {"rms_mean": float, "rms_std": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"rms_mean": np.nan, "rms_std": np.nan}
    
    rms_val = np.sqrt(np.mean(np.square(signal), dtype=np.float64))
    return {"rms_mean": float(rms_val), "rms_std": 0.0}