# src/features/spectral_entropy.py
import numpy as np
import librosa
from scipy.stats import entropy

def compute(signal: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 512) -> dict:
    """
    Berechnet die spektrale Entropie eines Audiosignals.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        n_fft (int): FFT-Länge (Default: 1024).
        hop_length (int): Schrittweite zwischen Frames (Default: 512).
    
    Returns:
        dict: {"entropy_mean": float, "entropy_std": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"entropy_mean": np.nan, "entropy_std": np.nan}

    # PSD berechnen
    psd = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))**2

    # Energie pro Frame
    frame_energy = np.sum(psd, axis=0)

    # Frames ohne Energie auf kleinen Wert setzen
    frame_energy[frame_energy == 0] = 1e-6

    # Normalisieren
    psd_norm = psd / frame_energy[None, :]

    # Entropie pro Frame
    spectral_entropy = entropy(psd_norm, axis=0)

    # NaNs abfangen
    spectral_entropy = np.nan_to_num(spectral_entropy, nan=0.0)

    return {
        "entropy_mean": float(np.mean(spectral_entropy)),
        "entropy_std":  float(np.std(spectral_entropy))
    }