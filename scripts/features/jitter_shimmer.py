# src/features/jitter_shimmer.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int, fmin: int = 75, fmax: int = 300) -> dict:
    """
    Berechnet Jitter (F0-Instabilität) und Shimmer (Amplituden-Instabilität).
    
    Quelle:
        - Titze, I. R. (1992). "Jitter, shimmer, and noise in voice."
        - Praat Manual
    
    Returns:
        dict: {"jitter_local": ..., "shimmer_local": ...}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"jitter_local": np.nan, "shimmer_local": np.nan}
    
    # F0-Schätzung pro Frame
    f0, voiced_flag, _ = librosa.pyin(signal, fmin=fmin, fmax=fmax, sr=sr)
    f0 = f0[~np.isnan(f0)]
    if f0.size < 2:
        return {"jitter_local": np.nan, "shimmer_local": np.nan}
    
    # Jitter: mittlere relative Abweichung der Periodendauer
    period = 1.0 / f0
    jitter_local = np.mean(np.abs(np.diff(period))) / np.mean(period)
    
    # Shimmer: mittlere relative Abweichung der Amplitude
    frames = librosa.util.frame(signal, frame_length=sr//100, hop_length=sr//200).T
    amplitudes = np.max(np.abs(frames), axis=1)
    shimmer_local = np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes)
    
    return {"jitter_local": float(jitter_local), "shimmer_local": float(shimmer_local)}