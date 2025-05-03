import numpy as np
from scipy.signal import lfilter
from scipy.linalg import toeplitz

def calculate_formant_bandwidths(audio_signal, sample_rate=16000, lpc_order=12, num_formants=3):
    """
    Berechnet die Bandbreiten der ersten Formantfrequenzen eines Audiosignals mit LPC-Analyse.

    Quelle:
        Markel, J. D., & Gray, A. H. (1976). Linear Prediction of Speech.
        Springer-Verlag.

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).
        sample_rate (int): Sampling-Rate des Signals (Standard: 16000 Hz).
        lpc_order (int): Ordnung des LPC-Modells (Standard: 12).
        num_formants (int): Anzahl der zu extrahierenden Formanten (Standard: 3).

    Returns:
        np.array: Die geschätzten Formant-Bandbreiten (in Hz), ohne negative Werte.
    """
    # Berechnung der Autokorrelation
    autocorr = np.correlate(audio_signal, audio_signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Nur positive Hälfte verwenden

    # Yule-Walker-Gleichung lösen, um LPC-Koeffizienten zu berechnen
    R = toeplitz(autocorr[:lpc_order])
    r = autocorr[1:lpc_order+1]
    
    try:
        lpc_coeffs = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        return np.array([])  # Falls Matrix singulär ist (z. B. Stille), keine Bandbreiten zurückgeben

    # Berechnung der Polstellen des LPC-Filters
    roots = np.roots(np.concatenate([[1], -lpc_coeffs]))  # LPC-Koeffizienten als Filter nutzen
    roots = [r for r in roots if np.imag(r) >= 0]  # Nur positive Frequenzen behalten

    # Umwandlung in Frequenzen & Bandbreiten
    formant_frequencies = np.angle(roots) * (sample_rate / (2 * np.pi))
    formant_bandwidths = -np.log(np.abs(roots)) * (sample_rate / (2 * np.pi))

    # Entferne unrealistische Werte (< 50 Hz) und sortiere nach Frequenz
    valid_indices = np.where(formant_frequencies > 50)[0]
    formant_frequencies = formant_frequencies[valid_indices]
    formant_bandwidths = np.abs(formant_bandwidths[valid_indices])  # 🚀 **Jetzt absolute Werte verwenden!**

    # Sortieren nach Frequenz und die ersten num_formants Formanten zurückgeben
    sorted_indices = np.argsort(formant_frequencies)
    formant_frequencies = formant_frequencies[sorted_indices][:num_formants]
    formant_bandwidths = formant_bandwidths[sorted_indices][:num_formants]

    return formant_bandwidths