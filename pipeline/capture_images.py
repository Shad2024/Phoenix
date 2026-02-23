from datetime import datetime
import cv2
import os
import time
#from voice import speak, start_listening
from pipeline.VoiceAssistant import speak, start_listening, stop


capturing = False
stop_capture = False

def capture_images( max_images=100,  delay=2):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join("../outputs/captured_images", timestamp)

    global capturing, stop_capture 
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        speak(" الكاميرا غير موجودة.")
        print(" Camera not found.")
        return None

    count = 0
    speak("الكاميرا جاهزة. قل 'ابدأ' للبدء بالتصوير.")
    print(" Say 'start capturing' to begin, 'stop capturing' to end.")

    def handle_voice_command(cmd):
        global capturing, stop_capture
        if "ابدأ " in cmd:
            if not capturing:
                speak("جارِ بداية التصوير.")
                capturing = True
                stop_capture = False
            else:
                speak("تم بدء التصوير بالفعل.")
        elif "stop "  in cmd:
            if capturing:
                speak("تم إيقاف التصوير.")
                stop_capture = True
            else:
                speak("لم يتم البدء بالتصوير بعد.")

    # Start listening in the background
    start_listening(handle_voice_command)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to capture frame.")
            break

        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

        if capturing and not stop_capture:
            filepath = os.path.join(output_folder, f"image_{count}.jpg")
            cv2.imwrite(filepath, frame)
            print(f" Captured {filepath}")
            count += 1
            if count >= max_images:
                speak("اكتمل التصوير.")
                break
            time.sleep(delay)

        if stop_capture:
            speak(f"تم الإيقاف. {count} صورة تم التقاطها.")
            break

    cap.release()
    cv2.destroyAllWindows()
    speak(f"تم حفظ جميع الصور في {output_folder}")
    stop()
    return output_folder


if __name__ == "__main__":
    folder = capture_images()
    print(f" All images saved to: {folder}")
