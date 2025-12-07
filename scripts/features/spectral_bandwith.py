# src/features/spectral_bandwidth.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> dict:
    """
    Berechnet die Spectral Bandwidth (Bandbreite des Spektrums) eines Audiosignals.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        frame_length (int): FFT-Länge (Default: 2048).
        hop_length (int): Schrittweite zwischen Frames in Samples (Default: 512).
    
    Returns:
        dict: {"bandwidth_mean": float, "bandwidth_std": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"bandwidth_mean": np.nan, "bandwidth_std": np.nan}

    # Frame-basierte Spectral Bandwidth (Shape: (1, n_frames))
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, n_fft=frame_length, hop_length=hop_length
    )[0]

    return {
        "bandwidth_mean": float(np.mean(bandwidth)),
        "bandwidth_std":  float(np.std(bandwidth))
    }