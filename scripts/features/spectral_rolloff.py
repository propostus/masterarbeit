# src/features/spectral_rolloff.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int, roll_percent: float = 0.85) -> dict:
    """
    Berechnet den Spectral Rolloff eines Audiosignals.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        roll_percent (float): Anteil der Energie, unterhalb dessen die Grenzfrequenz berechnet wird (Default: 0.85).
    
    Returns:
        dict: {"rolloff_mean": float, "rolloff_std": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"rolloff_mean": np.nan, "rolloff_std": np.nan}
    
    # Frame-basiert: Rolloff pro Frame in Hz
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=roll_percent)[0]
    
    return {
        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_std":  float(np.std(rolloff))
    }