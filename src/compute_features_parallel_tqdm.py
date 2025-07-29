import os
import pandas as pd
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Für Fortschrittsanzeige

from src.preprocessing import preprocess_audio

# Feature-Importe
from src.features.chroma_features import calculate_chroma_features
from src.features.clipping_ratio import calculate_clipping_ratio
from src.features.crest_factor import calculate_crest_factor
# from src.features.dnsmos import DNSMOS
# from src.features.formant_bandwith import calculate_formant_bandwidths
# from src.features.formant_freq import calculate_formants
from src.features.fundamental_freq import calculate_f0
from src.features.hnr import calculate_hnr
from src.features.log_energy import calculate_log_energy
from src.features.loudness_range import calculate_loudness_range
from src.features.mfcc import calculate_mfcc_spectrum, calculate_mfcc_statistics
from src.features.phoneme_entropy import calculate_phoneme_entropy
from src.features.reverberation import calculate_rt60
from src.features.rms import calculate_rms
from src.features.snr import calculate_snr
from src.features.spectral_bandwith import calculate_spectral_bandwidth
from src.features.spectral_centroid import calculate_spectral_centroid
from src.features.spectral_contrast import calculate_spectral_contrast
from src.features.spectral_entropy import calculate_spectral_entropy
from src.features.spectral_flatness import calculate_spectral_flatness
from src.features.spectral_rolloff import calculate_spectral_rolloff
from src.features.vad import calculate_vad
from src.features.zcr import calculate_zcr

# Parameter
audio_dir = "audio_files/common_voice_subset_50h/"
results_path = "results/subset_50h/features_250728.csv"
temp_path = "results/subset_50h/_tmp_features_parallel.csv"
sample_rate = 16000
save_interval = 100


def process_file(file_path, sample_rate=16000):
    try:
        audio_signal = preprocess_audio(file_path, sample_rate=sample_rate)
        filename = os.path.basename(file_path)

        features = {
            "filename": filename,
            "filepath": file_path,
            "rms": calculate_rms(audio_signal),
            "log_energy": calculate_log_energy(audio_signal),
            "clipping_ratio": calculate_clipping_ratio(audio_signal),
            "crest_factor": calculate_crest_factor(audio_signal),
            "snr": calculate_snr(audio_signal, sample_rate),
            "hnr": calculate_hnr(audio_signal, sample_rate),
            "f0": calculate_f0(audio_signal, sample_rate),
            "phoneme_entropy": calculate_phoneme_entropy(audio_signal, sample_rate),
            "rt60_reverberation": calculate_rt60(audio_signal, sample_rate),
            "spectral_bandwidth": calculate_spectral_bandwidth(audio_signal, sample_rate),
            "spectral_centroid": calculate_spectral_centroid(audio_signal, sample_rate),
            "spectral_contrast": calculate_spectral_contrast(audio_signal, sample_rate),
            "spectral_entropy": calculate_spectral_entropy(audio_signal, sample_rate),
            "spectral_flatness": calculate_spectral_flatness(audio_signal),
            "spectral_rolloff": calculate_spectral_rolloff(audio_signal, sample_rate),
            "vad": calculate_vad(audio_signal, sample_rate),
            "zcr": calculate_zcr(audio_signal),
            "loudness_range": calculate_loudness_range(audio_signal, lower_percentile=10, upper_percentile=95),
        }

        # Mehrdimensionale Features
        # formants = calculate_formants(audio_signal, sample_rate)
        # formant_bandwidths = calculate_formant_bandwidths(audio_signal, sample_rate)
        chroma = calculate_chroma_features(audio_signal, sample_rate)
        mfcc_spectrum = calculate_mfcc_spectrum(audio_signal, sample_rate)
        mfcc_stats = calculate_mfcc_statistics(mfcc_spectrum)
        # dnsmos_scores = dnsmos_model.calculate_dnsmos(audio_signal)

        features.update({
            # **{f"formant_{i+1}": val for i, val in enumerate(formants)},
            # **{f"formant_bw_{i+1}": val for i, val in enumerate(formant_bandwidths)},
            **{f"chroma_{i+1}": val for i, val in enumerate(chroma)},
            **{f"mfcc_stat_{i+1}": stat for i, stat in enumerate(mfcc_stats)},
            # **dnsmos_scores
        })

        return features

    except Exception as e:
        print(f"x Fehler bei Datei {file_path}: {e}")
        return None


def main():
    if os.path.exists(temp_path):
        os.remove(temp_path)

    file_list = [
        os.path.join(root, f)
        for root, _, files in os.walk(audio_dir)
        for f in files if f.endswith(".mp3")
    ]

    results = []

    with ProcessPoolExecutor() as executor:
        for i, result in enumerate(
            tqdm(executor.map(partial(process_file, sample_rate=sample_rate), file_list),
                 total=len(file_list)), 1):

            if result is not None:
                results.append(result)

            if i % save_interval == 0:
                df = pd.DataFrame(results)
                df.to_csv(temp_path, mode='a', header=not os.path.exists(temp_path), index=False)
                results.clear()
                print(f"- {i} Dateien verarbeitet und zwischengespeichert...")

    # Restliche Ergebnisse speichern
    if results:
        df = pd.DataFrame(results)
        df.to_csv(temp_path, mode='a', header=not os.path.exists(temp_path), index=False)

    # Zusammenführen und finale Datei erzeugen
    df_all = pd.read_csv(temp_path)
    df_all.to_csv(results_path, index=False)
    os.remove(temp_path)

    print(f"- Alle Dateien verarbeitet ({len(file_list)}). Ergebnisse gespeichert in: {results_path}")


if __name__ == '__main__':
    main()