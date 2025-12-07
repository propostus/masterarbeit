# src/features/spectral_centroid.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> dict:
    """
    Berechnet den Spectral Centroid (Schwerpunkt des Spektrums) eines Audiosignals.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        frame_length (int): FFT-Länge (Default: 2048).
        hop_length (int): Schrittweite zwischen Frames in Samples (Default: 512).
    
    Returns:
        dict: {"centroid_mean": float, "centroid_std": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"centroid_mean": np.nan, "centroid_std": np.nan}

    # Frame-basierter Spectral Centroid (Shape: (1, n_frames))
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

    return {
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std":  float(np.std(centroid))
    }