# src/features/hnr.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> dict:
    """
    Berechnet das Harmonics-to-Noise Ratio (HNR) basierend auf Autokorrelation (Praat-ähnlich).
    Quelle: Boersma (1993), "Accurate short-term analysis of the fundamental frequency..."

    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        frame_length (int): Framegröße in Samples (Default: 2048).
        hop_length (int): Schrittweite in Samples (Default: 512).

    Returns:
        dict: {"hnr_mean": float, "hnr_std": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"hnr_mean": np.nan, "hnr_std": np.nan}

    hnr_values = []

    # Framing
    for i in range(0, len(signal) - frame_length, hop_length):
        frame = signal[i:i + frame_length]
        if np.all(frame == 0):
            continue

        # Autokorrelation
        acf = np.correlate(frame, frame, mode="full")
        acf = acf[len(acf)//2:]  # nur positive Lags

        if np.max(acf) <= 0:
            continue

        # Normalisieren
        acf /= np.max(acf)

        # Peak nach Lag=0 suchen (Grundperiode)
        peak = np.max(acf[1:])  # höchster Peak nach 0-Lag
        if peak <= 1e-6:
            continue

        # HNR: Verhältnis peak/(1-peak) (Boersma)
        hnr = 10 * np.log10(peak / (1 - peak))
        hnr_values.append(hnr)

    if not hnr_values:
        return {"hnr_mean": np.nan, "hnr_std": np.nan}

    return {
        "hnr_mean": float(np.mean(hnr_values)),
        "hnr_std": float(np.std(hnr_values))
    }