# src/features/harmonic_ratio.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int) -> dict:
    """
    Schätzt das Verhältnis voiced/unvoiced Frames (Voicedness).
    
    Quelle:
        - Boersma, P. (1993). "Accurate short-term analysis of the fundamental frequency and the
          harmonics-to-noise ratio of a sampled sound." Proc. IFA.
    """
    if signal.size == 0:
        return {"harmonic_ratio": np.nan}
    
    f0, voiced_flag, _ = librosa.pyin(signal, fmin=75, fmax=300, sr=sr)
    if voiced_flag is None:
        return {"harmonic_ratio": np.nan}
    
    harmonic_ratio = np.mean(voiced_flag.astype(float))
    return {"harmonic_ratio": float(harmonic_ratio)}