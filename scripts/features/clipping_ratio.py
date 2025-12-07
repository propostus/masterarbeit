# src/features/clipping_ratio.py
import numpy as np

def compute(signal: np.ndarray, sr: int, threshold: float = 0.95) -> dict:
    """
    Berechnet die Clipping Ratio eines Audiosignals.
    
    Quelle:
        - ITU-R BS.1534 (MUSHRA): Clipping als Qualitätsbeeinträchtigung
        - Audio Engineering Society (AES) Guidelines zu Audioqualität

    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono, normalisiert -1..1).
        sr (int): Samplingrate in Hz (hier nicht genutzt, nur für API-Konsistenz).
        threshold (float): Clipping-Schwelle (Default: 0.95).

    Returns:
        dict: {"clipping_ratio": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"clipping_ratio": np.nan}

    clipped_samples = np.sum(np.abs(signal) >= threshold)
    clipping_ratio = clipped_samples / len(signal)

    return {"clipping_ratio": float(clipping_ratio)}