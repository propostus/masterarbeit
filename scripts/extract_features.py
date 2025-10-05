# scripts/extract_features.py
import os
import argparse
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Importiere Core-Features
from src.features import (
    rms,
    log_energy,
    clipping_ratio,
    crest_factor,
    zcr,
    vad,
    mfcc_deltas,
    chroma_features,
    plp,
    c50_c80,
    rt60,
    srmr,
    loudness_range,
    phoneme_entropy,
    snr_energy,
    snr_welch,
    spectral_centroid,
    spectral_rolloff,
    spectral_contrast,
    spectral_entropy,
    spectral_flatness,
    spectral_bandwith,
)

def preprocess_audio(y, sr, target_sr=16000, trim_db=30):
    """Standard-Preprocessing: Resample, Normalisierung, Trim Silence"""
    # Resample falls nötig
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Normalisierung auf [-1, 1]
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Leading/Trailing Silence entfernen
    y, _ = librosa.effects.trim(y, top_db=trim_db)

    return y

def extract_features_from_file(file_path, sample_rate=16000):
    """Extrahiert alle Core-Features aus einer Datei."""
    try:
        # Audio laden
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        y = preprocess_audio(y, sr, sample_rate)

        # Dict für Features
        feats = {"filename": os.path.basename(file_path)}

        # Features berechnen
        feats.update(rms.compute(y, sr))
        feats.update(log_energy.compute(y, sr))
        feats.update(clipping_ratio.compute(y, sr))
        feats.update(crest_factor.compute(y, sr))
        feats.update(zcr.compute(y, sr))
        feats.update(vad.compute(y, sr))
        feats.update(mfcc_deltas.compute(y, sr))
        feats.update(chroma_features.compute(y, sr))
        feats.update(plp.compute(y, sr))
        feats.update(c50_c80.compute(y, sr))
        feats.update(rt60.compute(y, sr))
        feats.update(srmr.compute(y, sr))
        feats.update(loudness_range.compute(y, sr))
        feats.update(phoneme_entropy.compute(y, sr))
        feats.update(snr_energy.compute(y, sr))
        feats.update(snr_welch.compute(y, sr))
        feats.update(spectral_centroid.compute(y, sr))
        feats.update(spectral_rolloff.compute(y, sr))
        feats.update(spectral_contrast.compute(y, sr))
        feats.update(spectral_entropy.compute(y, sr))
        feats.update(spectral_flatness.compute(y, sr))
        feats.update(spectral_bandwith.compute(y, sr))

        return feats
    except Exception as e:
        print(f"Fehler bei Datei {file_path}: {e}")
        return None

def main(audio_dir, out_csv, sample_rate=16000, max_files=None):
    # Alle Dateien im Verzeichnis finden
    audio_files = [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.lower().endswith((".wav", ".mp3", ".flac"))
    ]
    if max_files:
        audio_files = audio_files[:max_files]

    print(f"Extrahiere Features für {len(audio_files)} Dateien aus {audio_dir}")

    feature_list = []
    for f in tqdm(audio_files, desc="Feature extraction"):
        feats = extract_features_from_file(f, sample_rate)
        if feats:
            feature_list.append(feats)

    # DataFrame speichern
    df = pd.DataFrame(feature_list)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Features gespeichert unter: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True, help="Ordner mit Audiodateien (z. B. clips/)")
    parser.add_argument("--out_csv", type=str, required=True, help="Pfad zur Output-CSV")
    parser.add_argument("--sr", type=int, default=16000, help="Sampling Rate")
    parser.add_argument("--max_files", type=int, default=None, help="Maximale Anzahl an Dateien")
    args = parser.parse_args()

    main(args.audio_dir, args.out_csv, args.sr, args.max_files)