# src/features/log_energy.py
import numpy as np

def compute(signal: np.ndarray, sr: int) -> dict:
    """
    Berechnet die logarithmische Energie eines Audiosignals.
    (vereinfachte Version nach ETSI ES 201 108)

    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate (hier nicht genutzt, nur für API-Konsistenz).

    Returns:
        dict: {"log_energy_db": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"log_energy_db": np.nan}

    # Energie = Summe der Quadrate
    energy = np.sum(signal**2)

    if energy <= 0:
        return {"log_energy_db": np.nan}

    # Logarithmische Energie in dB
    log_energy_db = 10 * np.log10(energy + 2e-22)  # ETSI-Offset für Stabilität
    return {"log_energy_db": float(log_energy_db)}