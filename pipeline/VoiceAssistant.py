import uuid
import asyncio
import edge_tts
import re
import vosk
import json
import queue
import threading
import time
import tempfile
from playsound import playsound
import os
import sounddevice as sd
import soundfile as sf
from normalization_dict import NORMALIZATION_DICT

speech_queue = queue.Queue()
pause_listening = threading.Event()
sd.default.device = None  # input = default, output = speakers (WASAPI)


def normalize_cmd(text: str) -> str:
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)

    for wrong, correct in NORMALIZATION_DICT.items():
        pattern = rf'(?<!\S){re.escape(wrong)}(?!\S)'
        new_text = re.sub(pattern, correct, text)
        if new_text != text:
            #print(f"[DEBUG] Replaced '{wrong}' -> '{correct}' in '{text}'")
            text = new_text

    text = re.sub(r'\s+', ' ', text)
    return text

async def _speak_async(text, voice="ar-EG-SalmaNeural"):

    #tmp_file = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")
    #communicate = edge_tts.Communicate(text, voice=voice)
    #await communicate.save(tmp_file)
    #playsound(tmp_file)
    #os.remove(tmp_file)
    tmp_file = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.wav")

    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(tmp_file)

    data, samplerate = sf.read(tmp_file, dtype='float32')
    sd.play(data, samplerate)
    sd.wait()

    os.remove(tmp_file)

def tts_loop(voice="ar-EG-SalmaNeural"):

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            pause_listening.set()
            loop.run_until_complete(_speak_async(text, voice))
        except Exception as e:
            print("[TTS ERROR]", e)
        finally:
            pause_listening.clear()
            speech_queue.task_done()

    loop.close()

tts_thread = threading.Thread(target=tts_loop, daemon=True)
tts_thread.start()

def speak(text: str):
    """Queue text for speech without blocking the caller."""
    print(f"[TTS] {text}")
    speech_queue.put(text)


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vosk_model_path = os.path.join(base_dir, "models", "vosk-model-ar-mgb2-0.4")
asr_model = vosk.Model(vosk_model_path)
rec = vosk.KaldiRecognizer(asr_model, 16000)
listening = True

def listen_loop(callback, stop_phrase="وقف", chunk_duration=7):
    """
    Background listening loop. Calls callback when speech is recognized.
    """
    global listening
    import sounddevice as sd

    print("🎤 Continuous listening started...")

    while listening:
        if pause_listening.is_set():
            time.sleep(0.05)
            continue

        try:
            recording = sd.rec(int(chunk_duration * 16000), samplerate=16000, channels=1, dtype='int16')
            sd.wait()

            if rec.AcceptWaveform(recording.tobytes()):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text :
                    normalized_text = normalize_cmd(text)
                    print(f"[ASR] Detected: {text}")
                    print(f"[DEBUG] Normalized: {normalized_text}")
                    if stop_phrase in normalized_text:
                        print("Stop phrase detected. Stopping listening...")
                        listening = False
                        break
                    callback(normalized_text)

        except Exception as e:
            print("[LISTEN ERROR]", e)
            time.sleep(0.1)

def start_listening(callback, stop_phrase="وقف"):
    """Start listening in a background thread."""
    t = threading.Thread(target=listen_loop, args=(callback, stop_phrase), daemon=True)
    t.start()
    return t

def stop():
    """Stop TTS and listening threads cleanly."""
    global listening
    listening = False
    speech_queue.put(None)
    tts_thread.join()
    print("⛔ Offline server stopped.")
