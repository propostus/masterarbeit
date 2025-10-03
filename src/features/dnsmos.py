# src/features/dnsmos.py
import torch
import numpy as np
import librosa
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore

class DNSMOS:
    """
    DNSMOS (Deep Noise Suppression Mean Opinion Score).
    Nicht-intrusive Sprachqualitätsmetrik basierend auf einem Deep Learning Modell von Microsoft Research.
    
    Quelle:
        - Reddy et al., "DNSMOS: A Non-Intrusive Perceptual Objective Speech Quality Metric
          to Evaluate Noise Suppressors", arXiv:2010.15258 (2020).
        - https://github.com/microsoft/DNS-Challenge
    """

    def __init__(self, sample_rate: int = 16000, personalized: bool = False, device: str = "cpu"):
        self.sample_rate = sample_rate
        self.model = DeepNoiseSuppressionMeanOpinionScore(
            fs=sample_rate,
            personalized=personalized,
            device=device
        )

    def compute(self, signal: np.ndarray, sr: int) -> dict:
        """
        Berechnet DNSMOS-Scores für ein Audiosignal.

        Args:
            signal (np.ndarray): 1D-Audiosignal (float, mono).
            sr (int): Samplingrate des Signals in Hz.

        Returns:
            dict: {"p808_mos": float, "sig_mos": float, "bak_mos": float, "ovr_mos": float}
        """
        if signal.size == 0 or not np.isfinite(signal).any():
            return {"p808_mos": np.nan, "sig_mos": np.nan, "bak_mos": np.nan, "ovr_mos": np.nan}

        # Resampling falls notwendig
        if sr != self.sample_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)

        # Tensor erstellen
        audio_tensor = torch.tensor(signal).float()

        # DNSMOS Scores berechnen
        scores = self.model(audio_tensor)

        return {
            "p808_mos": scores[0].item(),
            "sig_mos": scores[1].item(),
            "bak_mos": scores[2].item(),
            "ovr_mos": scores[3].item()
        }