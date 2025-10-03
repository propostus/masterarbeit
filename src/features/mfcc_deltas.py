# src/features/mfcc_deltas.py
import numpy as np
import librosa
from scipy.stats import skew, kurtosis

def _extract_statistics(features: np.ndarray, prefix: str) -> dict:
    """Hilfsfunktion: berechnet Statistikwerte für ein Feature-Matrix (num_features x num_frames)."""
    stats = {}
    mean = np.mean(features, axis=1)
    std = np.std(features, axis=1)
    minv = np.min(features, axis=1)
    maxv = np.max(features, axis=1)
    rang = maxv - minv

    if np.all(np.isclose(features, features[:, 0][:, np.newaxis], atol=1e-8)):
        skewness = np.zeros(features.shape[0])
        kurt = np.zeros(features.shape[0])
    else:
        skewness = skew(features, axis=1)
        kurt = kurtosis(features, axis=1)

    for i in range(features.shape[0]):
        stats[f"{prefix}{i+1}_mean"] = float(mean[i])
        stats[f"{prefix}{i+1}_std"] = float(std[i])
        stats[f"{prefix}{i+1}_min"] = float(minv[i])
        stats[f"{prefix}{i+1}_max"] = float(maxv[i])
        stats[f"{prefix}{i+1}_range"] = float(rang[i])
        stats[f"{prefix}{i+1}_skew"] = float(skewness[i])
        stats[f"{prefix}{i+1}_kurtosis"] = float(kurt[i])

    return stats


def compute(signal: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512, num_mfcc: int = 13) -> dict:
    """
    Berechnet MFCCs + Delta-MFCCs + Delta-Delta-MFCCs inkl. Statistikwerte.

    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        frame_length (int): FFT-Fenstergröße.
        hop_length (int): Schrittweite zwischen Frames.
        num_mfcc (int): Anzahl der MFCC-Koeffizienten (Default: 13).

    Returns:
        dict: Statistikwerte für MFCC, Δ-MFCC und ΔΔ-MFCC.
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {f"{prefix}{i+1}_{stat}": np.nan
                for prefix in ["mfcc", "delta", "delta2"]
                for i in range(num_mfcc)
                for stat in ["mean", "std", "min", "max", "range", "skew", "kurtosis"]}

    # MFCC berechnen
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=num_mfcc,
        n_fft=frame_length,
        hop_length=hop_length
    )

    # Delta & Delta-Delta
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Statistiken extrahieren
    features = {}
    features.update(_extract_statistics(mfcc, "mfcc"))
    features.update(_extract_statistics(delta, "delta"))
    features.update(_extract_statistics(delta2, "delta2"))

    return features