import librosa
import numpy as np

def calculate_vad(audio_signal, sample_rate=16000, frame_length=1024, hop_length=512, threshold=0.05):
    """
    Berechnet den Prozentsatz der Sprachaktivität (VAD) in einem Audiosignal.

    Quelle:
        Maël Fabien, "Voice Activity Detection", 
        https://maelfabien.github.io/machinelearning/Speech4/#
        
    Args:
        audio_signal (np.array): Das normalisierte Audio-Signal (1D-Array).
        sample_rate (int): Sampling-Rate des Signals (Standard: 16000 Hz).
        frame_length (int): Länge eines Frames in Samples (Standard: 1024).
        hop_length (int): Schrittweite zwischen Frames in Samples (Standard: 512).
        threshold (float): Schwellenwert für die Energie, um Sprachaktivität zu erkennen.

    Returns:
        float: Prozentsatz der Sprachaktivität im Signal.
    """
    # Energie pro Frame berechnen
    energy = librosa.feature.rms(y=audio_signal, frame_length=frame_length, hop_length=hop_length)[0]

    # Sprachaktive Frames zählen (Energie > Threshold)
    num_active_frames = np.sum(energy > threshold)

    # Gesamtanzahl der Frames
    total_frames = len(energy)

    # Prozentsatz der Sprachaktivität
    vad_percentage = (num_active_frames / total_frames) * 100
    
    return vad_percentage