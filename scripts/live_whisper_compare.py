import sounddevice as sd
import numpy as np
import whisper
import threading
import time
import wave
import os
import warnings
from tabulate import tabulate

# Warnungen unterdrücken
warnings.filterwarnings("ignore")

# Aufnahmeparameter
MAX_DURATION = 10  # Sekunden
SAMPLERATE = 16000
CHANNELS = 1

# Zielverzeichnis für temporäres Audiofile
TEMP_DIR = "audio_files/temp"
os.makedirs(TEMP_DIR, exist_ok=True)
TEMP_FILE = os.path.join(TEMP_DIR, "last_recording.wav")

recording = []
is_recording = False


def record_audio():
    """Nimmt Audio vom Mikrofon auf, bis Enter gedrückt oder MAX_DURATION erreicht ist."""
    global recording, is_recording
    recording = []
    print("🎙️ Aufnahme läuft... (drücke Enter zum Stoppen)")
    start_time = time.time()
    with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype="float32") as stream:
        while is_recording and (time.time() - start_time < MAX_DURATION):
            audio_chunk, _ = stream.read(1024)
            recording.append(audio_chunk)
    print("✅ Aufnahme beendet.\n")


def save_audio(filename=TEMP_FILE):
    """Speichert die aufgenommene Audiodatei als WAV im TEMP_DIR."""
    if not recording:
        raise RuntimeError("Keine Aufnahme gefunden. Aufnahme muss gestartet werden.")
    audio = np.concatenate(recording, axis=0)
    scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
    with wave.open(filename, "w") as f:
        f.setnchannels(CHANNELS)
        f.setsampwidth(2)
        f.setframerate(SAMPLERATE)
        f.writeframes(scaled.tobytes())
    return filename


def main():
    global is_recording
    print("\nWhisper Live Vergleich – Modelle: Tiny, Base, Small, Medium, Large")
    print("-----------------------------------------------------------------\n")

    # Alte Aufnahme löschen (wenn vorhanden)
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

    input("Drücke Enter, um die Aufnahme zu starten...")
    is_recording = True
    rec_thread = threading.Thread(target=record_audio)
    rec_thread.start()
    input()  # Enter zum Stoppen
    is_recording = False
    rec_thread.join()

    wav_file = save_audio()
    print(f"🎧 Audio gespeichert unter: {wav_file}\n")

    models = [
        ("tiny", "tiny"),
        ("base", "base"),
        ("small", "small"),
        ("medium", "medium"),
        ("large", "large"),
    ]

    results = []
    for name, model_size in models:
        print(f"🧠 Transkribiere mit Whisper {name} ...")
        model = whisper.load_model(model_size)
        result = model.transcribe(wav_file, language="de")
        results.append([name, result["text"].strip()])

    print("\n📊 Vergleich der Transkriptionen:")
    print(tabulate(results, headers=["Modell", "Transkription"], tablefmt="grid"))


if __name__ == "__main__":
    main()