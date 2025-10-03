# src/features/spectro_temporal_entropy.py
import numpy as np
import librosa
from scipy.stats import entropy

def compute(signal: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 512) -> dict:
    """
    Kombinierte spektrale + zeitliche Entropie.
    
    Quelle:
        - H. Hermansky, "Perceptual linear predictive (PLP) analysis of speech," JASA, 1990.
    """
    if signal.size == 0:
        return {"spectro_temporal_entropy": np.nan}
    
    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))**2
    psd = S / np.sum(S, axis=0, keepdims=True)
    
    # spektrale Entropie pro Frame
    spec_entropy = entropy(psd, axis=0)
    # zeitliche Entropie über die Energiehüllkurve
    energy = np.sum(S, axis=0)
    prob_energy = energy / np.sum(energy)
    temp_entropy = entropy(prob_energy)
    
    return {
        "spectro_entropy_mean": float(np.mean(spec_entropy)),
        "temporal_entropy": float(temp_entropy)
    }