# src/features/vad.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int, frame_length: int = 1024, hop_length: int = 512, threshold: float = 0.05) -> dict:
    """
    Berechnet den Anteil sprachaktiver Frames (Voice Activity Detection, VAD) in einem Audiosignal.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz (Standard: 16000).
        frame_length (int): Länge eines Frames in Samples (Default: 1024).
        hop_length (int): Schrittweite in Samples (Default: 512).
        threshold (float): Energie-Schwellenwert für Sprachaktivität.
    
    Returns:
        dict: {"vad_ratio": float, "vad_num_active": int, "vad_total_frames": int}
              - vad_ratio = Anteil aktiver Frames (0.0–1.0)
              - vad_num_active = Anzahl aktiver Frames
              - vad_total_frames = Gesamtzahl Frames
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"vad_ratio": np.nan, "vad_num_active": 0, "vad_total_frames": 0}

    # Energie pro Frame berechnen
    energy = librosa.feature.rms(
        y=signal, frame_length=frame_length, hop_length=hop_length
    )[0]

    total_frames = len(energy)
    if total_frames == 0:
        return {"vad_ratio": np.nan, "vad_num_active": 0, "vad_total_frames": 0}

    # Sprachaktive Frames zählen
    num_active_frames = int(np.sum(energy > threshold))
    vad_ratio = num_active_frames / total_frames

    return {
        "vad_ratio": float(vad_ratio),
        "vad_num_active": num_active_frames,
        "vad_total_frames": total_frames,
    }