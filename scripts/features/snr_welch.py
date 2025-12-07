# src/features/snr_welch.py
import numpy as np
from scipy.signal import welch

def compute(signal: np.ndarray, sr: int, noise_duration: float = 0.5) -> dict:
    """
    Schätzt das Signal-Rausch-Verhältnis (SNR) mit der Welch-Methode.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        noise_duration (float): Länge des Noise-Segments (Sekunden, Default: 0.5).
    
    Returns:
        dict: {"snr_welch_db": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"snr_welch_db": np.nan}

    # Noise-Segment (Anfang des Signals)
    noise_end_sample = min(int(noise_duration * sr), len(signal))
    if noise_end_sample < 10:
        return {"snr_welch_db": np.nan}

    noise_segment = signal[:noise_end_sample]

    # PSD via Welch
    _, psd_signal = welch(signal, fs=sr, nperseg=1024)
    _, psd_noise = welch(noise_segment, fs=sr, nperseg=1024)

    signal_power = np.mean(psd_signal)
    noise_power = np.mean(psd_noise)

    if noise_power <= 1e-12:
        return {"snr_welch_db": 100.0}  # praktisch rauschfrei

    snr_value = 10 * np.log10(signal_power / noise_power)
    return {"snr_welch_db": float(snr_value)}