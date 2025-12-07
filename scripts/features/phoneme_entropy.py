# src/features/phoneme_entropy.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> dict:
    """
    Berechnet die spektrale Entropie als Proxy für Phonem-Variabilität ("Phoneme Entropy").
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        frame_length (int): FFT-Länge (Default: 2048).
        hop_length (int): Schrittweite zwischen Frames (Default: 512).
    
    Returns:
        dict: {"phoneme_entropy_mean": float, "phoneme_entropy_std": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {"phoneme_entropy_mean": np.nan, "phoneme_entropy_std": np.nan}

    # Power-Spektrogramm
    spectrogram = np.abs(librosa.stft(signal, n_fft=frame_length, hop_length=hop_length)) ** 2

    if np.sum(spectrogram) == 0:
        return {"phoneme_entropy_mean": np.nan, "phoneme_entropy_std": np.nan}

    # Normierung pro Frame → Verteilung über Frequenz
    prob_distribution = spectrogram / np.sum(spectrogram, axis=0, keepdims=True)

    # Shannon-Entropie pro Frame (log2)
    entropy_per_frame = -np.sum(prob_distribution * np.log2(prob_distribution + 1e-10), axis=0)

    return {
        "phoneme_entropy_mean": float(np.mean(entropy_per_frame)),
        "phoneme_entropy_std":  float(np.std(entropy_per_frame))
    }