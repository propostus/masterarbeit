# src/features/formants.py
import numpy as np
from scipy.linalg import toeplitz

def compute(signal: np.ndarray, sr: int, lpc_order: int = 12, num_formants: int = 3) -> dict:
    """
    Schätzt die Formantfrequenzen eines Audiosignals mit LPC-Analyse.
    
    Quelle:
        Markel, J. D., & Gray, A. H. (1976). Linear Prediction of Speech.
        Springer-Verlag.
    
    Args:
        signal (np.ndarray): 1D-Audiosignal (float, mono).
        sr (int): Samplingrate in Hz.
        lpc_order (int): Ordnung des LPC-Modells (Default: 12).
        num_formants (int): Anzahl der zu extrahierenden Formanten (Default: 3).
    
    Returns:
        dict: {"formant1_hz": float, "formant2_hz": float, "formant3_hz": float}
    """
    if signal.size == 0 or not np.isfinite(signal).any():
        return {f"formant{i+1}_hz": np.nan for i in range(num_formants)}

    # Autokorrelation
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr)//2:]

    if np.allclose(autocorr, 0):
        return {f"formant{i+1}_hz": np.nan for i in range(num_formants)}

    # Yule-Walker Gleichung für LPC
    try:
        R = toeplitz(autocorr[:lpc_order])
        r = autocorr[1:lpc_order+1]
        lpc_coeffs = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        return {f"formant{i+1}_hz": np.nan for i in range(num_formants)}

    # Polstellen
    roots = np.roots(np.concatenate([[1], -lpc_coeffs]))
    roots = [r for r in roots if np.imag(r) >= 0]

    # Frequenzen
    formant_frequencies = np.angle(roots) * (sr / (2 * np.pi))
    formant_frequencies = [f for f in formant_frequencies if f > 50]  # Filter unrealistische

    if len(formant_frequencies) == 0:
        return {f"formant{i+1}_hz": np.nan for i in range(num_formants)}

    formant_frequencies = np.sort(formant_frequencies)[:num_formants]

    # Dictionary mit formant1, formant2, ...
    return {
        f"formant{i+1}_hz": float(formant_frequencies[i]) if i < len(formant_frequencies) else np.nan
        for i in range(num_formants)
    }