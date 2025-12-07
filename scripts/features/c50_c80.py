# src/features/c50_c80.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int,
            frame_length: int = 1024,
            hop_length: int = 512) -> dict:
    """
    Proxy für Klarheitsmaße C50 und C80, berechnet aus der Energiehüllkurve.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        frame_length (int): RMS-Fenster (Default: 1024).
        hop_length (int): Schrittweite (Default: 512).
    
    Returns:
        dict: {"c50_proxy": float, "c80_proxy": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"c50_proxy": np.nan, "c80_proxy": np.nan}

    # Energiehüllkurve (RMS^2)
    energy_env = librosa.feature.rms(y=signal,
                                     frame_length=frame_length,
                                     hop_length=hop_length)[0]**2
    times = np.arange(len(energy_env)) * hop_length / sr

    def clarity_cut(ms):
        cutoff = ms / 1000.0
        early_energy = np.sum(energy_env[times <= cutoff])
        late_energy = np.sum(energy_env[times > cutoff])
        if late_energy <= 1e-12:
            return np.nan
        return 10 * np.log10(early_energy / late_energy)

    return {
        "c50_proxy": float(clarity_cut(50)),
        "c80_proxy": float(clarity_cut(80))
    }