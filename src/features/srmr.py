# src/features/srmr.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int,
            frame_length: int = 1024,
            hop_length: int = 512,
            low_band: tuple = (0.5, 4.0),
            speech_band: tuple = (4.0, 20.0)) -> dict:
    """
    SRMR-Proxy: Verhältnis von Sprach- zu Nachhall-bedingten Modulationsfrequenzen.
    Berechnet aus der RMS-Hüllkurve per FFT.

    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        frame_length (int): Fenstergröße für RMS-Berechnung.
        hop_length (int): Schrittweite für RMS-Berechnung.
        low_band (tuple): Nachhall-Bereich in Hz (Default: 0.5–4 Hz).
        speech_band (tuple): Sprach-Modulationen (Default: 4–20 Hz).

    Returns:
        dict: {"srmr_proxy": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"srmr_proxy": np.nan}

    # RMS-Hüllkurve als Approximation der Sprachmodulation
    rms_env = librosa.feature.rms(y=signal,
                                  frame_length=frame_length,
                                  hop_length=hop_length)[0]

    # Normalisieren
    rms_env = rms_env - np.mean(rms_env)
    if np.all(rms_env == 0):
        return {"srmr_proxy": np.nan}

    # FFT der Hüllkurve → Modulationsspektrum
    mod_spec = np.abs(np.fft.rfft(rms_env))**2
    freqs = np.fft.rfftfreq(len(rms_env), d=hop_length/sr)

    # Energie in Bändern summieren
    def band_energy(band):
        return np.sum(mod_spec[(freqs >= band[0]) & (freqs < band[1])])

    low_energy = band_energy(low_band)
    speech_energy = band_energy(speech_band)

    if low_energy <= 1e-12:
        return {"srmr_proxy": np.nan}

    return {"srmr_proxy": float(speech_energy / low_energy)}