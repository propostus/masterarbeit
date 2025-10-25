# scripts/add_noise_to_dataset.py
import os
import glob
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

def add_white_noise(audio, snr_db):
    """Add white noise at the given SNR (signal-to-noise ratio in dB)."""
    rms = np.sqrt(np.mean(audio ** 2))
    noise_rms = rms / (10 ** (snr_db / 20))
    noise = np.random.normal(0, noise_rms, audio.shape)
    return audio + noise

def process_files(input_dir, output_base, snr_levels, target_sr=32000, max_files=None):
    print(f"=== Erstelle verrauschte Versionen in {output_base} ===")
    os.makedirs(output_base, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.mp3")))
    if not files:
        print("Keine MP3-Dateien gefunden.")
        return
    if max_files:
        files = files[:max_files]

    for snr_db in snr_levels:
        out_dir = os.path.join(output_base, f"snr_{snr_db}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"→ Erstelle SNR {snr_db} dB: {out_dir}")

        for fpath in tqdm(files, desc=f"SNR {snr_db} dB", ncols=100):
            try:
                audio, sr = librosa.load(fpath, sr=None, mono=True)
                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_fast")
                noisy = add_white_noise(audio, snr_db)

                fname = os.path.splitext(os.path.basename(fpath))[0] + ".wav"
                out_path = os.path.join(out_dir, fname)
                sf.write(out_path, noisy, target_sr)
            except Exception as e:
                print(f"Fehler bei {fpath}: {e}")

    print("Fertig. Alle verrauschten Dateien wurden erstellt.")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Füge Rauschen zu Audiodateien hinzu (verschiedene SNR-Stufen).")
    parser.add_argument("--input_dir", type=str, required=True, help="Pfad zum Original-Audio-Ordner")
    parser.add_argument("--output_base", type=str, required=True, help="Zielordner für verrauschte Dateien")
    parser.add_argument("--snr_levels", nargs="+", type=float, default=[20, 10, 0], help="Liste der SNR-Stufen in dB")
    parser.add_argument("--target_sr", type=int, default=32000, help="Ziel-Samplingrate (Hz)")
    parser.add_argument("--max_files", type=int, default=None, help="Optionale Begrenzung der Dateien")

    args = parser.parse_args()
    process_files(
        input_dir=args.input_dir,
        output_base=args.output_base,
        snr_levels=args.snr_levels,
        target_sr=args.target_sr,
        max_files=args.max_files,
    )

if __name__ == "__main__":
    main()