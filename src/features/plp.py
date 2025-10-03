# src/features/plp.py
import numpy as np
import librosa

def compute(signal: np.ndarray, sr: int, num_coeffs: int = 13) -> dict:
    """
    Approximation von PLP-Features durch Bark-Filterbank + DCT.
    
    Quelle:
        - Hermansky, H. (1990). Perceptual linear predictive analysis of speech. JASA.
    """
    if signal.size == 0:
        return {f"plp{i+1}": np.nan for i in range(num_coeffs)}
    
    # Bark-Skala Filterbank
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=24)
    log_spec = librosa.power_to_db(mel_spec)
    dct = librosa.util.normalize(librosa.feature.mfcc(S=log_spec, sr=sr, n_mfcc=num_coeffs))
    
    return {f"plp{i+1}": float(np.mean(dct[i])) for i in range(num_coeffs)}