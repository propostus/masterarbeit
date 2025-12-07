# Prognose der Word Error Rate durch akustische Features
## Ein Machine-Learning-Ansatz zur Bewertung der Datenqualität für automatische Spracherkennung
### Master Thesis – Lukas Probst, 2025

Diese Arbeit untersucht die automatische Schätzung der Word Error Rate (WER) von Whisper ohne Referenztranskript.

Basierend auf SigMOS- und WavLM-Features werden Machine-Learning-Modelle eingesetzt, um:

- kontinuierliche WER-Schätzungen zu erzeugen (Regression)
- binäre Qualitätsschwellen zu klassifizieren:
  - WER unter 5 %?
  - WER unter 10 %?
  - WER unter 20 %?


## Repository-Struktur

masterarbeit/
├── src/                 # Anwendungscode (Features, Regression, Klassifikation)
├── models/              # Trainierte Modelle (Regression, Klassifikation) und SigMOS
├── requirements.txt     # Minimale Abhängigkeiten für die Anwendung
└── test/                # Beispielordner für eigene Audiodateien


## Voraussetzungen

- Python 3.11 (empfohlen)
- Conda
- macOS (getestet auf Apple Silicon)
- Ordner mit Sprachdateien in gängigen Formaten (.wav, .mp3, .flac)


## Installation

### 1. Repository klonen

git clone https://github.com/propostus/masterarbeit.git
cd masterarbeit

### 2. Conda-Environment erstellen und aktivieren

conda create -n wer-estimation python=3.11
conda activate wer-estimation

### 3. Abhängigkeiten installieren

pip install -r requirements.txt


## Vorbereitung

### Ergebnisordner anlegen

mkdir results

### Test-Audiodateien ablegen

masterarbeit/
    test/
        beispiel_1.wav
        beispiel_2.mp3


## Anwendung

Das zentrale Skript ist:

src/run_wer_estimation.py

Parameter:
--audio_dir
--mode  (regression | classification)
--device  (cpu | cuda | mps | auto)
--out_csv


## 1. Anwendungsbeispiel Regression – Kontinuierliche WER-Schätzung

python src/run_wer_estimation.py \
  --audio_dir test \
  --mode regression \
  --device auto \
  --out_csv results/test_regression.csv

Output-Spalten:
filename
filepath
wer_tiny # geschätzte WER, wenn diese Datei mit Whiper tiny transkribiert wird
wer_base # geschätzte WER, wenn diese Datei mit Whiper base transkribiert wird
wer_small # geschätzte WER, wenn diese Datei mit Whiper small transkribiert wird


## 2. Anwendungsbeispiel Klassifikation – WER-Schwellen 

python src/run_wer_estimation.py \
  --audio_dir test \
  --mode classification \
  --device auto \
  --out_csv results/test_classification.csv

Output-Spalten in der CSV-Datei (0 = WER überhalb Schwellwert, 1 = WER unterhalb Schwellwert):
wer_tiny_under_05_percent 
wer_tiny_under_10_percent
wer_tiny_under_20_percent
wer_base_under_05_percent
wer_base_under_10_percent
wer_base_under_20_percent
wer_small_under_05_percent
wer_small_under_10_percent
wer_small_under_20_percent


## Hardware / Performance

--device auto wählt automatisch:
  1. mps (Apple Silicon)
  2. cuda (NVIDIA)
  3. cpu


## Lizenz & Verwendung

Dieses Repository ist Teil der Masterarbeit von
Lukas Probst (2025), Fachgebiet Audiokommunikation, Technische Universität Berlin


## Kontakt

Fragen oder Feedback:
l.probst@campus.tu-berlin.de

