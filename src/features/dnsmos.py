import torch
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore

class DNSMOS:
    """
    DNSMOS (Deep Noise Suppression Mean Opinion Score) Modell zur Bewertung der Sprachqualität ohne Referenzsignal.
    
    Quelle:
        - Microsoft Research: DNSMOS – "Deep Noise Suppression Mean Opinion Score"
        - Reddy et al., "DNSMOS: A Non-Intrusive Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors", 
          in *arXiv preprint arXiv:2010.15258*, 2020.
        - https://github.com/microsoft/DNS-Challenge
    
    Args:
        sample_rate (int): Sampling-Rate des Audiosignals.
        personalized (bool): Ob das Modell störende Sprecher erkennt und in die Bewertung einbezieht.
        device (str): Rechen-Device (z. B. "cpu" oder "cuda").
    
    Returns:
        dict: DNSMOS-Scores für die Bewertung der Sprachqualität (P808_MOS, SIGMOS, BAKMOS, OVRMOS).
    """
    
    def __init__(self, sample_rate=16000, personalized=False, device="cpu"):
        self.model = DeepNoiseSuppressionMeanOpinionScore(fs=sample_rate, personalized=personalized, device=device)
    
    def calculate_dnsmos(self, audio_signal):
        """
        Berechnet die DNSMOS-Scores für ein Audio-Signal.
        
        Args:
            audio_signal (np.array): Vorverarbeitetes Audiosignal (Mono, 16 kHz, Normalisiert).
        
        Returns:
            dict: DNSMOS-Werte {P808_MOS, SIGMOS, BAKMOS, OVRMOS}.
        """
        # Umwandlung in Tensor für TorchMetrics
        audio_tensor = torch.tensor(audio_signal).float()

        # Berechnung der DNSMOS-Scores
        dnsmos_scores = self.model(audio_tensor)

        return {
            "P808_MOS": dnsmos_scores[0].item(),  # Perceptual MOS (Speech Quality)
            "SIGMOS": dnsmos_scores[1].item(),   # Speech Signal Quality
            "BAKMOS": dnsmos_scores[2].item(),   # Background Noise Quality
            "OVRMOS": dnsmos_scores[3].item()    # Overall Quality
        }