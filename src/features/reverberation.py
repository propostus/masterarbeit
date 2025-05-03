import numpy as np
import librosa

def calculate_rt60(audio_signal, sample_rate=16000):
    """
    Schätzt die Halligkeit (RT60) eines Audiosignals basierend auf der Schroeder-Rückwärtsintegration.

    Quelle:
        Schroeder, M. R. (1965). 
        "New Method of Measuring Reverberation Time." 
        *The Journal of the Acoustical Society of America*.

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).
        sample_rate (int): Sampling-Rate des Signals (Standard: 16000 Hz).

    Returns:
        float: RT60-Wert in Sekunden (oder NaN bei fehlender Schätzung).
    """
    # Berechne das Energieprofil (Squared Signal)
    energy = audio_signal ** 2

    # Falls das Signal zu leise ist, brechen wir ab
    if np.max(energy) == 0:
        return np.nan

    # Rückwärtsintegration (Schroeder-Integration)
    energy_reversed = np.cumsum(energy[::-1])[::-1]

    # Normalisiere die Energie (max = 0 dB)
    energy_db = 10 * np.log10(energy_reversed / np.max(energy_reversed) + 1e-10)  # Schutz vor log(0)

    # Finde -5 dB und -35 dB Punkte (anstatt -60 dB für stabilere Schätzung)
    idx_5dB = np.where(energy_db <= -5)[0]
    idx_35dB = np.where(energy_db <= -35)[0]

    # Falls keine Punkte gefunden wurden, RT60 als NaN zurückgeben
    if len(idx_5dB) == 0 or len(idx_35dB) == 0:
        return np.nan

    # Berechne die Zeitdifferenz
    t_5dB = idx_5dB[0] / sample_rate
    t_35dB = idx_35dB[0] / sample_rate

    # Lineare Regression zur RT60-Schätzung
    rt60 = (t_35dB - t_5dB) * (60 / 30)  # Hochrechnung von -30 dB auf -60 dB

    return rt60