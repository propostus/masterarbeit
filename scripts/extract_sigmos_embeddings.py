# scripts/extract_sigmos_embeddings.py
import os
import glob
import torch
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm
from models.sigmos.sigmos_model import SigMOSEstimator
from models.sigmos.compare_sigmos_stft import SignalProcessor


def extract_sigmos_embeddings(audio_dir, output_csv, device="cpu", max_files=None):
    print(f"=== Extrahiere SigMOS MOS-Werte von {audio_dir} ===")
    print(f"Verwende Gerät: {device}")

    # Audio-Backend für MP3 aktivieren
    torchaudio.set_audio_backend("sox_io")

    # Modell laden
    model = SigMOSEstimator().to(device)
    model.eval()
    processor = SignalProcessor()

    # Audiodateien suchen (mp3, flac, wav)
    audio_files = sorted(
        glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)
        + glob.glob(os.path.join(audio_dir, "**", "*.mp3"), recursive=True)
        + glob.glob(os.path.join(audio_dir, "**", "*.flac"), recursive=True)
    )

    if not audio_files:
        print("Keine Audiodateien gefunden.")
        return
    if max_files:
        audio_files = audio_files[:max_files]

    results = []

    for audio_path in tqdm(audio_files, desc="Verarbeite Dateien", ncols=100):
        try:
            # Audio laden
            waveform, sr = torchaudio.load(audio_path)
            audio = waveform.mean(dim=0).numpy()
            if sr != processor.sampling_rate:
                import librosa
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=processor.sampling_rate
                )

            # STFT und Feature-Kompression
            stft_result = processor.stft(audio)
            compressed = processor.compressed_mag_complex(stft_result)  # (1, 3, time, 481)
            x = torch.tensor(compressed, dtype=torch.float32).to(device)

            # Vorhersage
            with torch.no_grad():
                mos_vec = model(x).squeeze().cpu().numpy()
            mos_mean = float(np.mean(mos_vec))
            mos_std = float(np.std(mos_vec))

            results.append({
                "file": os.path.basename(audio_path),
                "mos_mean": mos_mean,
                "mos_std": mos_std,
                **{f"mos_{i+1}": mos_vec[i] for i in range(len(mos_vec))}
            })

        except Exception as e:
            print(f"Fehler bei {audio_path}: {e}")

    # Ergebnisse speichern
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(f"Fertig. Ergebnisse gespeichert unter: {output_csv}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extrahiere SigMOS MOS-Scores aus Audiodateien")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="audio_files/subsets/cv-corpus-de-combined-20-21-delta/clips",
        help="Pfad zum Ordner mit Audiodateien",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/embeddings/sigmos_results.csv",
        help="Pfad zur Ausgabedatei (CSV)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Rechengerät (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional: Begrenze die Anzahl verarbeiteter Dateien",
    )

    args = parser.parse_args()
    extract_sigmos_embeddings(
        audio_dir=args.audio_dir,
        output_csv=args.output_csv,
        device=args.device,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()