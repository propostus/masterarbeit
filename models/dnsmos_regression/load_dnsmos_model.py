import onnxruntime as ort
import numpy as np
import soundfile as sf
import os

# Pfad zum ONNX-Modell
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "sig_bak_ovr.onnx")

# Initialisiere ONNX Runtime Session
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Eingabe- und Ausgabe-Namen
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]


def extract_dnsmos_scores(file_path: str) -> dict:
    """
    Extrahiert SIG, BAK, OVR Scores aus einer Audiodatei mit dem DNSMOS ONNX-Modell.
    Args:
        file_path (str): Pfad zur Audiodatei (.wav, 16kHz, Mono)
    Returns:
        dict: Dictionary mit 'sig', 'bak', 'ovr'
    """
    # Audiodatei laden (muss 16kHz mono WAV sein!)
    audio, sr = sf.read(file_path)
    if sr != 16000:
        raise ValueError(f"Sampling rate must be 16kHz. Got {sr}")
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # Nur ersten Kanal nehmen

    # In ONNX-kompatibles Format bringen
    audio_tensor = audio.astype(np.float32)[np.newaxis, :]

    # Inferenz
    outputs = session.run(output_names, {input_name: audio_tensor})

    # Ausgabe-Parsing (reihenfolge: sig, bak, ovr)
    return {
        "sig": float(outputs[0].squeeze()),
        "bak": float(outputs[1].squeeze()),
        "ovr": float(outputs[2].squeeze())
    }