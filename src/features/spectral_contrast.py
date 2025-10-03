# src/features/spectral_contrast.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int, n_bands: int = 6) -> dict:
    """
    Berechnet den Spectral Contrast eines Audiosignals.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        n_bands (int): Anzahl der Frequenzbänder für die Kontrastanalyse (Default: 6).
    
    Returns:
        dict: z. B. {"contrast_band0_mean": ..., "contrast_band0_std": ..., ...}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {f"contrast_band{i}_mean": np.nan for i in range(n_bands+1)} | \
               {f"contrast_band{i}_std": np.nan for i in range(n_bands+1)}

    # Spektraler Kontrast: Shape (n_bands+1, n_frames)
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_bands=n_bands)

    features = {}
    for i in range(contrast.shape[0]):
        band_vals = contrast[i, :]
        features[f"contrast_band{i}_mean"] = float(np.mean(band_vals))
        features[f"contrast_band{i}_std"] = float(np.std(band_vals))

    return features