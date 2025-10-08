# scripts/extract_wavlm_embeddings.py
"""
Extrahiert Embeddings aus dem WavLM-Base-Modell für eine Sammlung von Audiodateien.
Erstellt eine CSV mit aggregierten Embeddings (mean + std pro Datei).
"""

import os
import argparse
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor


def extract_embedding(model, extractor, file_path, device):
    """Extrahiert den mean+std-Pooling-Embedding-Vektor aus einer Audiodatei."""
    try:
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0)  # mono
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
        hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        mean_emb = hidden.mean(axis=0)
        std_emb = hidden.std(axis=0)
        return np.concatenate([mean_emb, std_emb])
    except Exception as e:
        print(f"Fehler bei {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extrahiert WavLM-Embeddings aus Audiodateien.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Ordner mit Audiodateien (wav/mp3)")
    parser.add_argument("--out_csv", type=str, required=True, help="Pfad zur Ausgabe-CSV")
    parser.add_argument("--max_files", type=int, default=None, help="Maximale Anzahl Dateien (optional)")
    parser.add_argument("--device", type=str, default="cpu", help="Gerät: cpu, cuda oder mps")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA nicht verfügbar, verwende CPU.")
        device = torch.device("cpu")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device(args.device)
    print(f"Verwende Gerät: {device}")

    # Modell laden
    print("Lade WavLM Base-Modell...")
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = AutoModel.from_pretrained("microsoft/wavlm-base").to(device)
    model.eval()

    # Audiodateien auflisten
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    files = [os.path.join(args.audio_dir, f) for f in os.listdir(args.audio_dir) if f.lower().endswith(exts)]
    if args.max_files:
        files = files[: args.max_files]
    print(f"Verarbeite {len(files)} Dateien...\n")

    all_embs, all_names = [], []
    for f in tqdm(files, desc="Extrahiere Embeddings"):
        emb = extract_embedding(model, extractor, f, device)
        if emb is not None:
            all_embs.append(emb)
            all_names.append(os.path.basename(f))

    if not all_embs:
        print("Keine Embeddings extrahiert – überprüfe Pfad und Dateiformate.")
        return

    arr = np.vstack(all_embs)
    df = pd.DataFrame(arr)
    df.insert(0, "filename", all_names)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nFertig. Embeddings gespeichert unter: {args.out_csv}")


if __name__ == "__main__":
    main()