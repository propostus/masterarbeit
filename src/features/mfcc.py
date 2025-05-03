import numpy as np
import librosa
from scipy.stats import skew, kurtosis

def calculate_mfcc_spectrum(audio_signal, sample_rate=16000, frame_length=2048, hop_length=512, num_mfcc=13):
    """
    Berechnet das MFCC-Spektrum (rohe Koeffizienten).

    Args:
        audio_signal (np.array): Vorverarbeitetes Audiosignal.
        sample_rate (int): Sampling-Rate (Standard 16000 Hz).
        frame_length (int): FFT-Fenstergröße.
        hop_length (int): Schrittweite zwischen Frames.
        num_mfcc (int): Anzahl der MFCC-Koeffizienten.

    Returns:
        np.array: MFCC-Spektrum (num_mfcc x num_frames).
    """
    mfcc_features = librosa.feature.mfcc(
        y=audio_signal, 
        sr=sample_rate, 
        n_mfcc=num_mfcc, 
        n_fft=frame_length, 
        hop_length=hop_length
    )

    return mfcc_features

def calculate_mfcc_statistics(mfcc_features):
    """
    Berechnet Statistikwerte (mean, std, min, max, range, skewness, kurtosis) aus dem MFCC-Spektrum.

    Args:
        mfcc_features (np.array): MFCC-Spektrum (num_mfcc x num_frames).

    Returns:
        np.array: Feature-Vektor der MFCC-Statistiken (6 * num_mfcc Werte).
    """
    mfcc_mean = np.mean(mfcc_features, axis=1)
    mfcc_std = np.std(mfcc_features, axis=1)
    mfcc_min = np.min(mfcc_features, axis=1)
    mfcc_max = np.max(mfcc_features, axis=1)
    mfcc_range = mfcc_max - mfcc_min

    if np.all(np.isclose(mfcc_features, mfcc_features[:, 0][:, np.newaxis], atol=1e-8)):  
        mfcc_skewness = np.zeros(mfcc_features.shape[0])
        mfcc_kurtosis = np.zeros(mfcc_features.shape[0])
    else:
        mfcc_skewness = skew(mfcc_features, axis=1)
        mfcc_kurtosis = kurtosis(mfcc_features, axis=1)

    return np.concatenate([
        mfcc_mean, 
        mfcc_std, 
        mfcc_min, 
        mfcc_max, 
        mfcc_range, 
        mfcc_skewness, 
        mfcc_kurtosis
    ])