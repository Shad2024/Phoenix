
import pyttsx3
import speech_recognition as sr
import threading
import queue
import time

engine = pyttsx3.init(driverName="sapi5")
voices = engine.getProperty("voices")
for i, v in enumerate(voices):
    print("VOICE", i, v.id)
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate", 150)
speech_queue = queue.Queue()
pause_listening = threading.Event()


def tts_loop():
    """Background loop that handles all speech requests sequentially."""
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            pause_listening.set()
            time.sleep(0.08)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("[TTS ERROR]", e)
        finally:
            # Resume listening
            pause_listening.clear()
            speech_queue.task_done()



tts_thread = threading.Thread(target=tts_loop, daemon=True)
tts_thread.start()


def speak(text: str):
    """Queue text for speech without blocking the caller."""
    print(f"[TTS] {text}")
    speech_queue.put(text)


listening = True


def listen_loop(callback, stop_phrase="stop listening", timeout=20, phrase_time_limit=5):
    """Continuously listen for speech and call callback with recognized text."""
    global listening
    r = sr.Recognizer()
    print("🎤 Continuous listening started...")

    while listening:
        if pause_listening.is_set():
            time.sleep(0.05)
            continue

        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            # Recognize speech after releasing the mic
            response = r.recognize_google(audio).lower().strip()
            print(f"You said: {response}")

            if stop_phrase in response:
                print("Stop command detected. Stopping listening...")
                listening = False
                break

            # Pass recognized text to callback
            callback(response)

        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand.")
            continue
        except sr.RequestError as e:
            print("Speech Recognition service error:")
            break
        except Exception as e:
            print("[LISTEN ERROR]", e)
            time.sleep(0.1)



def start_listening(callback):
    """Start listening in a background thread."""
    t = threading.Thread(target=listen_loop, args=(callback,), daemon=True)
    t.start()
    return t


def stop_tts():
    """Stop TTS thread cleanly (optional cleanup)."""
    speech_queue.put(None)
    tts_thread.join()
