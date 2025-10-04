import numpy as np
import librosa

def compute(audio_signal: np.ndarray, sample_rate: int = 16000,
            n_fft: int = 1024, hop_length: int = 512) -> dict:
    """
    Berechnet die mittleren Chroma-Features eines Audiosignals.

    Quelle:
        - Librosa: Chroma Features (Tonhöhenverteilung)
        - https://librosa.org/doc/main/generated/librosa.feature.chroma_stft.html

    Args:
        audio_signal (np.array): Vorverarbeitetes Audio-Signal.
        sample_rate (int): Sampling-Rate des Audiosignals (Hz).
        n_fft (int): Anzahl der FFT-Punkte für die Spektralanalyse.
        hop_length (int): Schrittweite zwischen FFT-Berechnungen.

    Returns:
        dict: Mittlere Chroma-Werte pro Klasse { "chroma_0": ..., ..., "chroma_11": ... }
    """
    try:
        # Prüfen, ob Signal verwertbar ist
        if np.all(audio_signal == 0) or np.max(np.abs(audio_signal)) < 1e-6:
            raise ValueError("Signal ist leer oder zu leise.")

        chroma = librosa.feature.chroma_stft(
            y=audio_signal,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )

        # Prüfen auf ungültige Werte
        if np.any(np.isnan(chroma)) or np.any(np.isinf(chroma)):
            raise ValueError("Ungültige Werte in Chroma-Features.")

        # Mittelwert über die Zeitachse
        chroma_mean = np.mean(chroma, axis=1)

    except Exception as e:
        print(f"⚠️ Fehler in chroma_features.compute: {e}")
        chroma_mean = np.zeros(12)

    # Dictionary mit eindeutigen Keys zurückgeben
    return {f"chroma_{i}": float(chroma_mean[i]) for i in range(12)}