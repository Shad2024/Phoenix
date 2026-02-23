from VoiceAssistant import speak, start_listening, stop
from normalization_dict import NORMALIZATION_DICT
from Ara_recommend_materials import run_recommendation_model
import subprocess
import threading
import speech_recognition as sr
import os
import sys
import json


def normalize_command(command):
    if not command:
        return ""
    normalized_cmd = command.strip()

    for key, value in NORMALIZATION_DICT.items():
        if key in normalized_cmd:
            normalized_cmd = normalized_cmd.replace(key, value)
            
    return normalized_cmd


def handle_voice_command(cmd):
    cmd = normalize_command(cmd)
    cmd = cmd.lower()
    print(f"[DEBUG] Heard (normalized): {cmd}")

    base_dir = os.path.dirname(__file__)
    python_exe = os.path.join(base_dir, "..", ".venv", "Scripts", "python.exe")
    python_exe = os.path.normpath(python_exe)

    # Capture images module
    if ("ابدأ" in cmd and "التصوير" in cmd) or ("start capturing" in cmd and "camera" in cmd):
        speak("تشغيل وحدة التقاط الصور.")
        target = os.path.join(base_dir, "capture_images.py")
        subprocess.Popen([python_exe, target], cwd=base_dir)

    # Damage analysis module
    elif ("التحليل" in cmd and "الأضرار" in cmd) or ("start analysis" in cmd and "damage" in cmd):
        speak("  بدء تحليل الأضرار.")
        target = os.path.join(base_dir, "analyze_images.py")
        subprocess.Popen([python_exe, target], cwd=base_dir)

    # Material analysis module
    elif ("التحليل" in cmd and "المواد" in cmd) or ("start analysis" in cmd and "materials" in cmd):
        speak("بدء تحليل المواد.")
        target = os.path.join(base_dir, "Waste_Material.py")
        subprocess.Popen([python_exe, target], cwd=base_dir)

        # After it finishes, run your recommendation model
        target_recommendation = os.path.join(base_dir, "Ara_recommend_materials.py")
        subprocess.run([python_exe, target_recommendation], cwd=base_dir)
        run_recommendation_model()
        

        # Draw map module
    elif ("ارسم" in cmd and "خريطة" in cmd) or ("draw" in cmd and "map" in cmd):
        speak("جاري رسم الخريطة.")
        target = os.path.join(base_dir, "B_Map.py")
        subprocess.run([sys.executable, target], cwd=base_dir)


    # Stop everything
    elif "توقف" in cmd or "stop" in cmd:
        speak("تم إيقاف المشروع.")
        print(" Stopping project...")
        exit(0)


def get_command():
    """Ask the user to choose between voice or text input."""
    choice = input(
        "🎙️ اختر الوضع (v للصوت / t للنص / exit للخروج): ").strip().lower()

    if choice == "exit":
        speak("تم إيقاف المساعد.")
        print(" تم الإنهاء.")
        stop()
        exit(0)

    elif choice == "v":
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print(" استمع... تحدث الآن.")
            audio = recognizer.listen(source, phrase_time_limit=8)

        try:
            cmd = recognizer.recognize_google(audio, language="ar-SA")
            print(f"🗣️ Detected: {cmd}")
            return cmd
        except sr.UnknownValueError:
            print(" لم أفهم الصوت، حاول مجدداً.")
            return None
        except sr.RequestError:
            print(" خدمة التعرف على الصوت غير متاحة.")
            return None

    elif choice == "t":
        cmd = input("⌨️ أدخل الأمر النصي: ").strip()
        return cmd if cmd else None

    else:
        print(" خيار غير معروف. اكتب 'v' أو 't' أو 'exit'.")
        return None

def main():
    speak("تم تفعيل فنيكس.")
    print("Project activated in Arabic.")

    while True:
        cmd = get_command()      # 1️⃣ get STRING
        if cmd is None:
            continue

        handle_voice_command(cmd)  # 2️⃣ pass STRING

    while True:
        threading.Event().wait(1) 

if __name__ == "__main__":
    main()