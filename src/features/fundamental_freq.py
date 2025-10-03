# src/features/f0.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int,
            frame_length: int = 2048,
            hop_length: int = 512,
            fmin: int = 75,
            fmax: int = 300) -> dict:
    """
    Berechnet die Fundamental Frequency (F0) eines Audiosignals mit der pYIN-Methode.
    
    Quelle:
        Mauch, M., & Dixon, S. (2014). 
        pYIN: A fundamental frequency estimator using probabilistic threshold distributions.
        ICASSP 2014.

    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        frame_length (int): Länge eines FFT-Fensters (Default: 2048).
        hop_length (int): Schrittweite in Samples (Default: 512).
        fmin (int): minimale erwartete F0 (Default: 75 Hz).
        fmax (int): maximale erwartete F0 (Default: 300 Hz).

    Returns:
        dict: {"f0_mean_hz": float, "f0_std_hz": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"f0_mean_hz": np.nan, "f0_std_hz": np.nan}

    try:
        f0_values, voiced_flag, _ = librosa.pyin(
            signal, fmin=fmin, fmax=fmax, sr=sr,
            frame_length=frame_length, hop_length=hop_length
        )
    except Exception:
        return {"f0_mean_hz": np.nan, "f0_std_hz": np.nan}

    if f0_values is None:
        return {"f0_mean_hz": np.nan, "f0_std_hz": np.nan}

    # Nur voiced Frames behalten
    f0_values = f0_values[~np.isnan(f0_values)]
    if f0_values.size == 0:
        return {"f0_mean_hz": np.nan, "f0_std_hz": np.nan}

    return {
        "f0_mean_hz": float(np.mean(f0_values)),
        "f0_std_hz":  float(np.std(f0_values))
    }