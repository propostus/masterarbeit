# src/features/rt60.py
import numpy as np
import librosa

def compute(signal: np.ndarray,
            sr: int,
            frame_length: int = 1024,
            hop_length:   int = 512,
            energy_threshold: float = 0.02,
            tail_seconds: float = 1.0) -> dict:
    """
    Schätzt RT60 aus einem Nachhall-Tail hinter der letzten Sprachaktivität.
    1) Sprachaktivität via RMS-Schwelle
    2) Tail ausschneiden
    3) Schroeder-Rückwärtsintegration
    4) lineares Fitten im Bereich -5..-35 dB
    5) RT60 = -60 / slope

    Returns:
        dict: {"rt60_s": float|nan, "rt60_method": "schroeder_tail", "tail_duration_s": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"rt60_s": np.nan, "rt60_method": "schroeder_tail", "tail_duration_s": 0.0}

    # 1) grobe Sprachaktivität über RMS-Frames
    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0 or not np.isfinite(rms).any():
        return {"rt60_s": np.nan, "rt60_method": "schroeder_tail", "tail_duration_s": 0.0}

    # Schwelle (fix oder adaptiv). Hier: max(energy_threshold, 10%-Quantil)
    thr = max(energy_threshold, float(np.quantile(rms[np.isfinite(rms)], 0.10)))
    active_idx = np.where(rms > thr)[0]
    if active_idx.size == 0:
        return {"rt60_s": np.nan, "rt60_method": "schroeder_tail", "tail_duration_s": 0.0}

    # 2) Tail: ab Ende der letzten Aktivität
    last_frame = int(active_idx[-1])
    tail_start = (last_frame + 1) * hop_length
    tail_end   = min(len(signal), tail_start + int(tail_seconds * sr))
    if tail_end - tail_start < int(0.2 * sr):  # mind. 200 ms
        return {"rt60_s": np.nan, "rt60_method": "schroeder_tail", "tail_duration_s": (tail_end - tail_start) / sr}

    tail = signal[tail_start:tail_end]

    # 3) Schroeder-Rückwärtsintegration auf Energie
    e = tail.astype(np.float64) ** 2
    if np.max(e) <= 0:
        return {"rt60_s": np.nan, "rt60_method": "schroeder_tail", "tail_duration_s": (tail_end - tail_start) / sr}

    edc = np.cumsum(e[::-1])[::-1]           # Energy Decay Curve
    edc /= np.max(edc)                        # normieren
    edc_db = 10.0 * np.log10(edc + 1e-12)     # in dB, 0 dB = Start

    # 4) lineares Fitten im Bereich -5..-35 dB
    t = np.arange(edc_db.size) / sr
    mask = (edc_db <= -5.0) & (edc_db >= -35.0)
    if np.count_nonzero(mask) < 5:  # zu wenig Punkte
        return {"rt60_s": np.nan, "rt60_method": "schroeder_tail", "tail_duration_s": (tail_end - tail_start) / sr}

    slope, intercept = np.polyfit(t[mask], edc_db[mask], 1)  # dB = slope * t + intercept
    if slope >= -1e-6:  # keine fallende Kurve
        return {"rt60_s": np.nan, "rt60_method": "schroeder_tail", "tail_duration_s": (tail_end - tail_start) / sr}

    rt60 = -60.0 / slope  # dB/s → Sekunden
    return {"rt60_s": float(rt60), "rt60_method": "schroeder_tail", "tail_duration_s": (tail_end - tail_start) / sr}