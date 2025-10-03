# src/features/zcr.py
import numpy as np

def compute(signal: np.ndarray, sr: int, frame_size: int = 1024, hop_length: int = 512) -> dict:
    """
    Berechnet die Zero Crossing Rate (ZCR) eines Audiosignals.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz (hier nicht zwingend benötigt).
        frame_size (int): Fenstergröße in Samples (Default: 1024).
        hop_length (int): Schrittweite in Samples (Default: 512).
    
    Returns:
        dict: {"zcr_mean": float, "zcr_std": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"zcr_mean": np.nan, "zcr_std": np.nan}

    # Padding am Ende, damit auch letztes Fenster reinpasst
    num_frames = 1 + (len(signal) - frame_size) // hop_length if len(signal) >= frame_size else 0
    zcr_per_frame = []

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_size
        frame = signal[start:end]
        # Zero crossings: Vorzeichenwechsel zählen
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        zcr_per_frame.append(zcr)

    if len(zcr_per_frame) == 0:
        return {"zcr_mean": np.nan, "zcr_std": np.nan}

    return {
        "zcr_mean": float(np.mean(zcr_per_frame)),
        "zcr_std":  float(np.std(zcr_per_frame))
    }