import os
import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

# --- NEU: Repo-Root auf sys.path legen, damit "models" gefunden wird ---
import sys
from pathlib import Path

# Pfad zum Projekt-Root: .../masterarbeit
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# SigMOS-Imports (Pfad wie in deinem Repo)
from models.sigmos.sigmos_model import SigMOSEstimator
from models.sigmos.compare_sigmos_stft import SignalProcessor

# WavLM-Imports
from transformers import AutoModel, AutoFeatureExtractor

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")  # optional

# ------------------------------------------------------------
# Gerät & Dateisuche
# ------------------------------------------------------------
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def resolve_device(device: str = "auto") -> torch.device:
    """
    Gibt einen sinnvollen torch.device zurück basierend auf dem String.
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def list_audio_files(audio_dir: str, recursive: bool = True) -> List[str]:
    """
    Sammelt alle Audio-Dateien in einem Ordner.
    """
    patterns = []
    if recursive:
        for ext in AUDIO_EXTS:
            patterns.append(os.path.join(audio_dir, "**", f"*{ext}"))
    else:
        for ext in AUDIO_EXTS:
            patterns.append(os.path.join(audio_dir, f"*{ext}"))

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=recursive))

    files = sorted(set(files))
    return files


# ------------------------------------------------------------
# SigMOS
# ------------------------------------------------------------
def load_sigmos_model(device: torch.device) -> Tuple[SigMOSEstimator, SignalProcessor]:
    """
    Lädt SigMOS-Modell + SignalProcessor.
    """
        # suppress backend warning just for this call
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        torchaudio.set_audio_backend("sox_io")
    model = SigMOSEstimator().to(device)
    model.eval()
    processor = SignalProcessor()
    return model, processor


def compute_sigmos_for_waveform(
    waveform: np.ndarray,
    sr: int,
    model: SigMOSEstimator,
    processor: SignalProcessor,
    device: torch.device,
) -> Tuple[np.ndarray, float, float]:
    """
    Berechnet SigMOS-Vector + mean + std für ein 1D-Waveform (np.array).
    """
    # ggf. Resampling
    if sr != processor.sampling_rate:
        import librosa
        waveform = librosa.resample(
            waveform, orig_sr=sr, target_sr=processor.sampling_rate
        )

    # STFT + Kompression
    stft_result = processor.stft(waveform)
    compressed = processor.compressed_mag_complex(stft_result)  # (1, 3, time, 481)
    x = torch.tensor(compressed, dtype=torch.float32).to(device)

    with torch.no_grad():
        mos_vec = model(x).squeeze().cpu().numpy()

    mos_mean = float(np.mean(mos_vec))
    mos_std = float(np.std(mos_vec))
    return mos_vec, mos_mean, mos_std


# ------------------------------------------------------------
# WavLM
# ------------------------------------------------------------
def load_wavlm_base(device: torch.device):
    """
    Lädt WavLM-Base Model + FeatureExtractor.
    """
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = AutoModel.from_pretrained("microsoft/wavlm-base").to(device)
    model.eval()
    return model, extractor


def compute_wavlm_embedding_for_waveform(
    waveform: torch.Tensor,
    sr: int,
    model,
    extractor,
    device: torch.device,
) -> np.ndarray:
    """
    Berechnet WavLM-Embedding (mean + std concatenated) für ein 1D-Tensor-Waveform.
    Rückgabe: np.array mit Shape (2 * hidden_dim,) (bei Base: 1536).
    """
    # auf Mono sicherstellen
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = extractor(
        waveform, sampling_rate=16000, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})

    hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    mean_emb = hidden.mean(axis=0)
    std_emb = hidden.std(axis=0)
    emb = np.concatenate([mean_emb, std_emb])
    return emb


# ------------------------------------------------------------
# High-Level Feature-Wrapper
# ------------------------------------------------------------
def extract_sigmos_and_wavlm_features(
    audio_dir: str,
    device: str = "auto",
    max_files: Optional[int] = None,
    include_paths: bool = True,
) -> pd.DataFrame:
    """
    Extrahiert SigMOS- und WavLM-Features für alle Audiodateien in einem Ordner
    und gibt ein DataFrame mit einer Zeile pro Datei zurück.

    Spalten:
      - filename
      - optional: filepath
      - mos_mean, mos_std, mos_1 ... mos_7
      - "0" ... "1535" (WavLM-Features, wie im Training)
    """
    dev = resolve_device(device)
    print(f"[Features] Verwende Gerät: {dev}")

    files = list_audio_files(audio_dir, recursive=False)
    if not files:
        print(f"[Features] Keine Audiodateien in {audio_dir} gefunden.")
        return pd.DataFrame()

    if max_files is not None:
        files = files[:max_files]

    print(f"[Features] Verarbeite {len(files)} Dateien aus {audio_dir}")

    # Modelle laden
    print("[Features] Lade SigMOS...")
    sigmos_model, sigmos_proc = load_sigmos_model(dev)

    print("[Features] Lade WavLM-Base...")
    wavlm_model, wavlm_extractor = load_wavlm_base(dev)

    rows = []

    for path in tqdm(files, desc="Feature-Extraktion", ncols=100):
        try:
            # Audio laden
            waveform, sr = torchaudio.load(path)
            # für SigMOS: numpy mono
            waveform_np = waveform.mean(dim=0).numpy()

            # SigMOS
            mos_vec, mos_mean, mos_std = compute_sigmos_for_waveform(
                waveform_np, sr, sigmos_model, sigmos_proc, dev
            )

            # WavLM (nutzt torch.Tensor)
            wavlm_emb = compute_wavlm_embedding_for_waveform(
                waveform, sr, wavlm_model, wavlm_extractor, dev
            )

            row = {}
            row["filename"] = os.path.basename(path)
            if include_paths:
                row["filepath"] = path

            # SigMOS-Features benennen: exakt wie im Training
            row["mos_mean"] = mos_mean
            row["mos_std"] = mos_std
            for i, v in enumerate(mos_vec):
                # mos_1 ... mos_7
                row[f"mos_{i+1}"] = float(v)

            # WavLM-Features benennen: Spalten "0" ... "1535"
            for i, v in enumerate(wavlm_emb):
                row[str(i)] = float(v)

            rows.append(row)

        except Exception as e:
            print(f"[Features] Fehler bei {path}: {e}")

    df = pd.DataFrame(rows)
    return df