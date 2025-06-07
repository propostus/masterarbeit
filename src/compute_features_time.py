import os
import pandas as pd
import time
from src.preprocessing import preprocess_audio

# Feature-Importe
from src.features.chroma_features import calculate_chroma_features
from src.features.clipping_ratio import calculate_clipping_ratio
from src.features.crest_factor import calculate_crest_factor
from src.features.dnsmos import DNSMOS
from src.features.formant_bandwith import calculate_formant_bandwidths
from src.features.formant_freq import calculate_formants
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
audio_dir = "audio_files/common_voice_test/"
timing_results_path = "results/feature_times.csv"
sample_rate = 16000

dnsmos_model = DNSMOS(sample_rate=sample_rate, personalized=False, device="cpu")

timing_results = []

for root, _, files in os.walk(audio_dir):
    for filename in files:
        if filename.endswith(".mp3"):
            file_path = os.path.join(root, filename)

            try:
                audio_signal = preprocess_audio(file_path, sample_rate=sample_rate)

                row = {"filename": filename, "filepath": file_path}

                # Dauer der analysierten Audiofiles
                duration_sec = len(audio_signal) / sample_rate
                row["duration_sec"] = round(duration_sec, 2)

                def time_it(name, func, *args, **kwargs):
                    start = time.perf_counter()
                    func(*args, **kwargs)
                    duration = time.perf_counter() - start
                    row[f"time_{name}"] = round(duration, 4)

                # Zeitmessung je Feature
                time_it("rms", calculate_rms, audio_signal)
                time_it("log_energy", calculate_log_energy, audio_signal)
                time_it("clipping_ratio", calculate_clipping_ratio, audio_signal)
                time_it("crest_factor", calculate_crest_factor, audio_signal)
                time_it("snr", calculate_snr, audio_signal, sample_rate)
                time_it("hnr", calculate_hnr, audio_signal, sample_rate)
                time_it("f0", calculate_f0, audio_signal, sample_rate)
                time_it("phoneme_entropy", calculate_phoneme_entropy, audio_signal, sample_rate)
                time_it("rt60_reverberation", calculate_rt60, audio_signal, sample_rate)
                time_it("spectral_bandwidth", calculate_spectral_bandwidth, audio_signal, sample_rate)
                time_it("spectral_centroid", calculate_spectral_centroid, audio_signal, sample_rate)
                time_it("spectral_contrast", calculate_spectral_contrast, audio_signal, sample_rate)
                time_it("spectral_entropy", calculate_spectral_entropy, audio_signal, sample_rate)
                time_it("spectral_flatness", calculate_spectral_flatness, audio_signal)
                time_it("spectral_rolloff", calculate_spectral_rolloff, audio_signal, sample_rate)
                time_it("vad", calculate_vad, audio_signal, sample_rate)
                time_it("zcr", calculate_zcr, audio_signal)
                time_it("loudness_range", calculate_loudness_range, audio_signal, 10, 95)
                time_it("formants", calculate_formants, audio_signal, sample_rate)
                time_it("formant_bandwidths", calculate_formant_bandwidths, audio_signal, sample_rate)
                time_it("chroma", calculate_chroma_features, audio_signal, sample_rate)
                time_it("mfcc_spectrum", calculate_mfcc_spectrum, audio_signal, sample_rate)
                time_it("mfcc_statistics", calculate_mfcc_statistics,    calculate_mfcc_spectrum(audio_signal, sample_rate))
                time_it("dnsmos", dnsmos_model.calculate_dnsmos, audio_signal)

                timing_results.append(row)

            except Exception as e:
                print(f"xxx Fehler bei Datei {file_path}: {e}")
                continue

# DataFrame erstellen und speichern
df = pd.DataFrame(timing_results)
df.to_csv(timing_results_path, index=False)
print(f"Zeitmessung abgeschlossen. Ergebnisse gespeichert in: {timing_results_path}")