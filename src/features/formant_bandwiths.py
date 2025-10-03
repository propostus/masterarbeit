# src/features/formant_bandwidths.py
import numpy as np
from scipy.linalg import toeplitz

def compute(signal: np.ndarray, sr: int, lpc_order: int = 12, num_formants: int = 3) -> dict:
    """
    Schätzt die Bandbreiten der ersten Formanten eines Audiosignals mit LPC-Analyse.
    
    Quelle:
        Markel, J. D., & Gray, A. H. (1976). Linear Prediction of Speech.
        Springer-Verlag.

    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        lpc_order (int): Ordnung des LPC-Modells (Default: 12).
        num_formants (int): Anzahl der zu extrahierenden Formanten (Default: 3).
    
    Returns:
        dict: {"formant1_bw_hz": float, "formant2_bw_hz": float, ...}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {f"formant{i+1}_bw_hz": np.nan for i in range(num_formants)}

    # Autokorrelation
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr)//2:]

    if np.allclose(autocorr, 0):
        return {f"formant{i+1}_bw_hz": np.nan for i in range(num_formants)}

    # Yule-Walker Gleichung lösen
    try:
        R = toeplitz(autocorr[:lpc_order])
        r = autocorr[1:lpc_order+1]
        lpc_coeffs = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        return {f"formant{i+1}_bw_hz": np.nan for i in range(num_formants)}

    # Polstellen des LPC-Filters
    roots = np.roots(np.concatenate([[1], -lpc_coeffs]))
    roots = [r for r in roots if np.imag(r) >= 0]

    if len(roots) == 0:
        return {f"formant{i+1}_bw_hz": np.nan for i in range(num_formants)}

    # Frequenzen und Bandbreiten
    freqs = np.angle(roots) * (sr / (2 * np.pi))
    bws = -np.log(np.abs(roots)) * (sr / (2 * np.pi))

    # Nur plausible Formanten (F > 50 Hz)
    valid_idx = np.where(freqs > 50)[0]
    freqs = freqs[valid_idx]
    bws = bws[valid_idx]

    if freqs.size == 0:
        return {f"formant{i+1}_bw_hz": np.nan for i in range(num_formants)}

    # Nach Frequenz sortieren
    sort_idx = np.argsort(freqs)
    bws = bws[sort_idx][:num_formants]

    # Rückgabe als Dict
    return {
        f"formant{i+1}_bw_hz": float(bws[i]) if i < len(bws) else np.nan
        for i in range(num_formants)
    }