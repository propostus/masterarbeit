import numpy as np
from scipy.signal import welch

def calculate_snr(audio_signal, sample_rate=16000, noise_duration=0.5):
    """
    Schätzt das Signal-Rausch-Verhältnis (SNR) basierend auf FFT und spektraler Analyse.

    Quelle:
        Harris, F., & Dick, C. (2012). 
        SNR Estimation Techniques for Low SNR Signals. 
        In *The 15th International Symposium on Wireless Personal Multimedia Communications* 
        (pp. 24-27). IEEE Xplore. 
        DOI: 10.1109/WPMC.2012.6370441

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).
        sample_rate (int): Sampling-Rate des Signals (Standard: 16000 Hz).
        noise_duration (float): Länge der Rauschschätzphase in Sekunden (Standard: 0.5s).

    Returns:
        float: Signal-to-Noise Ratio (SNR) in dB.
    """
    # Anzahl der Samples für die Noise-Region
    noise_end_sample = min(int(noise_duration * sample_rate), len(audio_signal))
    noise_segment = audio_signal[:noise_end_sample]

    # Berechne PSD mit Welch's Methode
    f_signal, psd_signal = welch(audio_signal, fs=sample_rate, nperseg=1024)
    f_noise, psd_noise = welch(noise_segment, fs=sample_rate, nperseg=1024)

    # Mittlere Leistung des Signals und des Rauschens
    signal_power = np.mean(psd_signal)
    noise_power = np.mean(psd_noise)

    # Verhindere Division durch Null
    if noise_power == 0:
        return 100.0  # Bedeutet praktisch kein Rauschen

    # Berechnung des SNR in dB
    snr_value = 10 * np.log10(signal_power / noise_power)

    return snr_value