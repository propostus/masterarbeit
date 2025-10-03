# src/features/spectral_flatness.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> dict:
    """
    Berechnet die Spectral Flatness eines Audiosignals.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz (wird nicht direkt benötigt, nur für Konsistenz).
        frame_length (int): Länge eines Frames in Samples (Default: 2048).
        hop_length (int): Schrittweite zwischen Frames in Samples (Default: 512).
    
    Returns:
        dict: {"flatness_mean": float, "flatness_std": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"flatness_mean": np.nan, "flatness_std": np.nan}

    # Flatness pro Frame (Werte zwischen 0 und 1)
    flatness = librosa.feature.spectral_flatness(y=signal, n_fft=frame_length, hop_length=hop_length)[0]

    return {
        "flatness_mean": float(np.mean(flatness)),
        "flatness_std":  float(np.std(flatness))
    }