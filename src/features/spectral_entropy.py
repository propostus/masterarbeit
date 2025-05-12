import librosa
import numpy as np
from scipy.stats import entropy

def calculate_spectral_entropy(audio_signal, sample_rate=16000, n_fft=1024, hop_length=512):
    """
    Berechnet die spektrale Entropie eines Audiosignals.

    Quelle:
        - Konzept von Entropy: Shannon Entropy (Informationsgehalt)
        - Librosa: Spektrale Leistungsdichte (PSD)
        - https://librosa.org/doc/main/generated/librosa.feature.spectral_flatness.html

    Args:
        audio_signal (np.array): Vorverarbeitetes Audio-Signal.
        sample_rate (int): Sampling-Rate des Audiosignals.
        n_fft (int): Anzahl der FFT-Punkte für die Spektralanalyse.
        hop_length (int): Schrittweite zwischen FFT-Berechnungen.

    Returns:
        float: Durchschnittliche spektrale Entropie des Signals oder 0 bei Stille.
    """
    # Berechnung des Leistungsdichtespektrums (PSD)
    psd = np.abs(librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length))**2

    # Prüfe, ob das gesamte Signal still ist (keine Energie)
    total_energy = np.sum(psd)
    if total_energy < 1e-6:
        return 0.0  # Kein nennenswerter Informationsgehalt, daher Entropie = 0

    # Prüfe, ob es Frames mit Energie = 0 gibt
    frame_energy = np.sum(psd, axis=0)
    
    #if np.any(frame_energy == 0):
        #print(f"Warnung: Frames mit 0 Energie gefunden! Werte werden korrigiert. Anzahl: {np.sum(frame_energy == 0)}")

    # Falls ein Frame 0 Energie hat, setzen wir ihn auf einen kleinen Wert (1e-6)
    frame_energy[frame_energy == 0] = 1e-6

    # Normalisierung mit stabiler Mindestgrenze
    psd_norm = psd / frame_energy[None, :]

    # Prüfe auf NaN-Werte in der Normalisierung
    #if np.any(np.isnan(psd_norm)):
        #print("NaN-Warnung: NaN-Werte in der PSD-Normalisierung!")

    # Berechnung der Entropie pro Frame
    spectral_entropy = entropy(psd_norm, axis=0)

    # Falls nach der Berechnung noch NaN-Werte existieren, setze sie auf 0
    if np.any(np.isnan(spectral_entropy)):
        #print("NaN-Warnung: NaN-Werte in der Entropieberechnung! Setze auf 0.")
        spectral_entropy = np.nan_to_num(spectral_entropy, nan=0.0)

    return np.mean(spectral_entropy)