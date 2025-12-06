# scripts/augment_with_esc50.py
"""
Erzeugt zu jedem Clean-File eine zusätzlich augmentierte Version durch Addieren
von realistischen Geräuschen aus ESC-50 bei zufälliger SNR im (hohen) Bereich.

- Audio wird auf 16 kHz, mono normiert.
- Pro Clean-File wird 1 Noise-Sample zufällig gewählt (inkl. zufälligem Ausschnitt).
- Ziel-SNR wird gleichverteilt in [--snr_min, --snr_max] dB gezogen (z. B. 15..30 dB).
- Ergebnis-Datei bekommt Suffix _aug.wav und wird in --out_dir geschrieben.
- Es entsteht eine augmentation_log.csv mit allen Parametern.

Beispiel:
python -m scripts.augment_with_esc50 \
  --clean_dir audio_files/subsets/cv-corpus-de-combined-20-21-delta/clips \
  --esc50_dir audio_files/noise/ESC-50/audio \
  --out_dir audio_files/augmented/esc50_snr15_30 \
  --snr_min 15 --snr_max 30 --seed 42
"""

import os
import glob
import math
import random
import argparse
import numpy as np
import pandas as pd
import torchaudio
import torchaudio.functional as AF
from tqdm import tqdm

TARGET_SR = 16000

def load_mono_16k(path):
    wav, sr = torchaudio.load(path)  # (ch, n)
    wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)  # (n,)

def rms(x: np.ndarray):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))

def apply_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Skaliert noise so, dass mix eine Ziel-SNR (clean/noise) in dB hat.
    """
    # Angleichen der Längen: zufälliger Ausschnitt oder zyklisch wiederholen
    if len(noise) >= len(clean):
        start = np.random.randint(0, len(noise) - len(clean) + 1)
        noise_seg = noise[start:start+len(clean)]
    else:
        reps = math.ceil(len(clean) / len(noise))
        noise_seg = np.tile(noise, reps)[:len(clean)]

    px = rms(clean)
    pn = rms(noise_seg)
    if pn < 1e-12:  # degenerativer Fall
        return clean.copy()

    # SNR_db = 10 log10(Px / Pn_scaled) => Pn_scaled = Px / 10^(SNR/10)
    # Skalierungsfaktor a für noise_seg: a^2 * pn^2 = (px^2) / 10^(SNR/10)
    target_noise_rms = px / (10.0**(snr_db/20.0))
    scale = target_noise_rms / pn
    mixed = clean + scale * noise_seg

    # Clipping-Gegenmaßnahme: leichtes Peak-Normieren, falls nötig
    peak = np.max(np.abs(mixed))
    if peak > 0.999:
        mixed = mixed / peak * 0.999
    return mixed

def main():
    p = argparse.ArgumentParser(description="Erzeuge augmentierte Kopien mit ESC-50 Noise")
    p.add_argument("--clean_dir", required=True, help="Ordner mit Clean-Audiodateien (wav/mp3/flac)")
    p.add_argument("--esc50_dir", required=True, help="Ordner mit ESC-50 Geräuschdateien (meist .../ESC-50/audio)")
    p.add_argument("--out_dir", required=True, help="Zielordner für augmentierte Dateien")
    p.add_argument("--snr_min", type=float, default=15.0, help="min. SNR in dB (höher = sauberer)")
    p.add_argument("--snr_max", type=float, default=30.0, help="max. SNR in dB")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torchaudio.set_audio_backend("sox_io")

    os.makedirs(args.out_dir, exist_ok=True)

    # Clean-Dateien sammeln
    exts = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    clean_files = []
    for ext in exts:
        clean_files.extend(glob.glob(os.path.join(args.clean_dir, ext)))
    clean_files = sorted(clean_files)

    # ESC-50 Geräuschdateien sammeln (ESC-50/audio enthält WAVs)
    noise_files = sorted(glob.glob(os.path.join(args.esc50_dir, "**", "*.wav"), recursive=True))
    if not noise_files:
        raise RuntimeError("Keine Noise-Dateien gefunden. Prüfe --esc50_dir (sollte auf ESC-50/audio zeigen).")

    print(f"Clean: {len(clean_files)} Dateien | Noise: {len(noise_files)} Dateien")
    log_rows = []

    for cf in tqdm(clean_files, desc="Augmentiere"):
        try:
            clean_wav = load_mono_16k(cf).numpy()
            noise_path = random.choice(noise_files)
            noise_wav = load_mono_16k(noise_path).numpy()

            snr_db = float(np.random.uniform(args.snr_min, args.snr_max))
            mixed = apply_snr(clean_wav, noise_wav, snr_db)

            base = os.path.splitext(os.path.basename(cf))[0]
            out_path = os.path.join(args.out_dir, f"{base}_aug.wav")
            torchaudio.save(out_path, torch.from_numpy(mixed).unsqueeze(0), TARGET_SR)

            log_rows.append({
                "clean_file": os.path.relpath(cf, args.clean_dir),
                "augmented_file": os.path.basename(out_path),
                "noise_file": os.path.relpath(noise_path, args.esc50_dir),
                "snr_db": snr_db
            })
        except Exception as e:
            print(f"Fehler bei {cf}: {e}")

    # Log speichern
    if log_rows:
        pd.DataFrame(log_rows).to_csv(os.path.join(args.out_dir, "augmentation_log.csv"), index=False)
        print(f"Fertig. Augmentierte Dateien: {len(log_rows)}")
        print(f"Log: {os.path.join(args.out_dir, 'augmentation_log.csv')}")
    else:
        print("Keine Dateien augmentiert.")

if __name__ == "__main__":
    import torch
    main()