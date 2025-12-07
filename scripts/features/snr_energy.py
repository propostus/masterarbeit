# src/features/snr_energy.py
import numpy as np

def compute(signal: np.ndarray, sr: int, noise_duration: float = 0.5) -> dict:
    """
    Schätzt das Signal-Rausch-Verhältnis (SNR) mit einem einfachen Energie-basierten Ansatz.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        noise_duration (float): Länge des Noise-Segments (Sekunden, Default: 0.5).
    
    Returns:
        dict: {"snr_energy_db": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"snr_energy_db": np.nan}

    # Noise-Segment (Anfang)
    noise_end_sample = min(int(noise_duration * sr), len(signal))
    if noise_end_sample < 10:
        return {"snr_energy_db": np.nan}

    noise_segment = signal[:noise_end_sample]

    # Energie = mittlere Quadratsumme
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise_segment**2)

    if noise_power <= 1e-12:
        return {"snr_energy_db": 100.0}  # praktisch rauschfrei

    snr_value = 10 * np.log10(signal_power / noise_power)
    return {"snr_energy_db": float(snr_value)}