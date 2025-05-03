import numpy as np
import librosa
from scipy.signal import find_peaks

def calculate_hnr_praat(audio_signal, sample_rate=16000, frame_length=2048, hop_length=512):
    """
    Berechnet das Harmonics-to-Noise Ratio (HNR) eines Audiosignals nach der Methode aus Praat (Hoole, Sprachproduktion 1).

    Quelle:
        Hoole, P. (Seminar Sprachproduktion 1, Akustische Analyse der Stimmqualität).
        Klatt & Klatt (1990), "Analysis, synthesis, and perception of voice quality variations".

    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).
        sample_rate (int): Sampling-Rate des Signals (Standard: 16000 Hz).
        frame_length (int): Länge eines Frames in Samples (Standard: 2048 für bessere Frequenzauflösung).
        hop_length (int): Schrittweite zwischen Frames in Samples (Standard: 512).

    Returns:
        float: Durchschnittlicher HNR-Wert des Signals in Dezibel (dB).
    """
    # Anzahl der Frames
    num_frames = (len(audio_signal) - frame_length) // hop_length + 1
    hnr_values = []

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio_signal[start:end]

        # FFT berechnen
        fft_spectrum = np.abs(np.fft.rfft(frame))
        fft_frequencies = np.fft.rfftfreq(len(frame), d=1/sample_rate)

        # Finde den Grundton (f0) - stärkster Peak im unteren Frequenzbereich
        peaks, properties = find_peaks(fft_spectrum, height=np.max(fft_spectrum) * 0.1)
        if len(peaks) < 2:
            hnr_values.append(0.0)  # Falls keine harmonischen Peaks gefunden werden
            continue  

        # Berechnung von H1-H2
        h1_index = peaks[0]  # Erste Harmonische (Grundton)
        h2_index = peaks[1]  # Zweite Harmonische
        h1_amplitude = properties["peak_heights"][0]
        h2_amplitude = properties["peak_heights"][1]
        h1_h2 = 10 * np.log10(h1_amplitude / h2_amplitude)  # Spektrale Neigung

        # Gesamtenergie des Spektrums
        total_energy = np.sum(fft_spectrum ** 2)

        # Harmonische Energie (nur Frequenzen unter 4 kHz verwenden)
        harmonic_energy = np.sum(fft_spectrum[fft_frequencies < 4000] ** 2)

        # Rauschenergie = Gesamtenergie - harmonische Energie
        noise_energy = total_energy - harmonic_energy

        # Fehlerbehandlung
        if noise_energy <= 0 or harmonic_energy <= 0:
            hnr = 0.0
        else:
            hnr = 10 * np.log10(harmonic_energy / noise_energy)

        hnr_values.append(hnr)

    # Durchschnittlicher HNR-Wert über alle Frames
    average_hnr = np.mean(hnr_values) if hnr_values else 0.0
    return average_hnr