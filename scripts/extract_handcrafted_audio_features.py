# scripts/extract_handcrafted_audio_features.py
# ------------------------------------------------------
# Extrahiert einfache, hand-crafted Audiofeatures:
#   - duration_s        (Länge in Sekunden)
#   - rms_mean          (mittlere Kurzzeit-RMS-Energie)
#   - rms_std           (Std der Kurzzeit-RMS-Energie)
#   - pause_ratio       (Anteil "stiller" Frames, Speaking-Rate-Proxy)
#   - pitch_mean        (mittlere Grundfrequenz, via YIN)
#   - pitch_std         (Std der Grundfrequenz)
#   - centroid_mean     (mittlerer Spektralzentroid)
#   - centroid_std      (Std des Spektralzentroids)
#
# Ausgabe: CSV mit Spalten:
#   filename, duration_s, rms_mean, rms_std,
#   pause_ratio, pitch_mean, pitch_std,
#   centroid_mean, centroid_std
# ------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from tqdm import tqdm


def compute_handcrafted_features(
    file_path,
    sr_target=16000,
    frame_length_ms=25,
    hop_length_ms=10,
    rms_silence_thresh_db=-40.0,
):
    """
    Berechnet Dauer, RMS-Statistiken, Pause-Ratio, Pitch-Statistiken
    und Spektralzentroid-Statistiken für eine Audiodatei.
    """

    try:
        waveform, sr = torchaudio.load(file_path)  # (channels, samples)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # In numpy umwandeln
        x = waveform.numpy().astype(np.float32)

        # Resampling auf sr_target
        if sr != sr_target:
            x = librosa.resample(x, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

        num_samples = x.shape[0]
        if num_samples == 0:
            return None

        duration_s = float(num_samples / sr)

        # Frame-Parameter
        frame_length = int(sr * frame_length_ms / 1000.0)
        hop_length = int(sr * hop_length_ms / 1000.0)
        frame_length = max(frame_length, 1)
        hop_length = max(hop_length, 1)

        # Kurzzeit-RMS
        rms = librosa.feature.rms(
            y=x,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]  # shape (T,)

        rms_mean = float(rms.mean())
        rms_std = float(rms.std()) if rms.size > 1 else 0.0

        # Speaking-Rate-Proxy: Anteil "stiller" Frames
        rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=1.0)
        silent_frames = rms_db < rms_silence_thresh_db
        pause_ratio = float(silent_frames.mean()) if rms.size > 0 else 0.0

        # Pitch-Statistiken (YIN)
        try:
            f0 = librosa.yin(
                x,
                fmin=80.0,
                fmax=400.0,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            f0 = f0[np.isfinite(f0)]
            if f0.size > 0:
                pitch_mean = float(f0.mean())
                pitch_std = float(f0.std())
            else:
                pitch_mean = np.nan
                pitch_std = np.nan
        except Exception:
            pitch_mean = np.nan
            pitch_std = np.nan

        # Spektralzentroid
        try:
            centroid = librosa.feature.spectral_centroid(
                y=x,
                sr=sr,
                n_fft=frame_length,
                hop_length=hop_length,
            )[0]
            centroid_mean = float(centroid.mean())
            centroid_std = float(centroid.std()) if centroid.size > 1 else 0.0
        except Exception:
            centroid_mean = np.nan
            centroid_std = np.nan

        return {
            "duration_s": duration_s,
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "pause_ratio": pause_ratio,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "centroid_mean": centroid_mean,
            "centroid_std": centroid_std,
        }

    except Exception as e:
        print(f"Fehler bei {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extrahiere hand-crafted Audiofeatures (Duration, RMS, Pause-Ratio, Pitch, Spectral Centroid)"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Ordner mit Audiodateien (wav/mp3/flac/ogg/m4a)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Pfad zur Ausgabedatei (CSV)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional: maximale Anzahl Dateien (Debug/Test)",
    )
    args = parser.parse_args()

    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    files = [
        os.path.join(args.audio_dir, f)
        for f in os.listdir(args.audio_dir)
        if f.lower().endswith(exts)
    ]
    files = sorted(files)
    if args.max_files is not None:
        files = files[: args.max_files]

    print(f"Finde {len(files)} Audiodateien in {args.audio_dir}")

    records = []
    for f in tqdm(files, desc="Berechne Features"):
        feats = compute_handcrafted_features(f)
        if feats is None:
            continue

        record = {"filename": os.path.basename(f)}
        record.update(feats)
        records.append(record)

    if not records:
        print("Keine Features berechnet.")
        return

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Fertig. Features gespeichert unter: {args.out_csv}")
    print(df.head())


if __name__ == "__main__":
    main()