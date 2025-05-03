import numpy as np

def calculate_zcr(audio_signal, frame_size=1024):
    """
    Berechnet die Zero Crossing Rate (ZCR) eines Audiosignals.

    Quelle:
        Tom Bäckström et al., "Introduction to Speech Processing", 2nd Edition, 2022.
        DOI: 10.5281/zenodo.6821775
        URL: https://speechprocessingbook.aalto.fi/Representations/Zero-crossing_rate.html

    Args:
        audio_signal (np.array): Das Audio-Signal (1D-Array).
        frame_size (int): Die Anzahl der Samples pro Frame.

    Returns:
        float: Durchschnittliche Zero Crossing Rate des Signals.
    """
    # Anzahl der Frames
    num_frames = len(audio_signal) // frame_size
    frames = audio_signal[:num_frames * frame_size].reshape((num_frames, frame_size))

    # ZCR pro Frame berechnen
    zcr_per_frame = []
    for frame in frames:
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        zcr_per_frame.append(zcr)

    # Durchschnittliche ZCR
    return np.mean(zcr_per_frame)