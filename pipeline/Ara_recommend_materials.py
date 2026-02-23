import threading
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import speech_recognition as sr
import time
import json
from transformers import pipeline
import random
from VoiceAssistant import speak, start_listening, stop, pause_listening

model_path = r"C:\Users\bolaky\PycharmProjects\Rebuild\Models\Ara-QA"
base_tokenizer_name = "asafaya/bert-base-arabic"
recognizer = sr.Recognizer()
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only= True).to(device)

with open(r"C:\Users\bolaky\PycharmProjects\Rebuild\Datasets\context_dict.json", "r", encoding="utf-8") as f:
    context_dict = json.load(f)

# Track which context index was used last for each material
#context_index_tracker = {material: 0 for material in context_dict.keys()}

last_material_discussed = None 



def find_material_in_question(question: str):
    """Find the material mentioned in the question."""
    for material in context_dict.keys():
        if material in question:
            return material
    return None

def handle_simple_yes_no(question: str, material: str):
    """
    Handles simple questions like 'Is it recyclable?' or 'What is its category?' 
    by extracting the answer directly from the first context sentence.
    """
    context_sentence = random.choice(context_dict[material]) 

    if "قابل لإعادة التدوير" in question or "هل يمكن اعادة تدوير" in question:
        if "قابلة لإعادة التدوير" in context_sentence or "قابل لإعادة التدوير" in context_sentence: 
            return "نعم، المادة قابلة لإعادة التدوير."
        elif "غير قابلة لإعادة التدوير" in context_sentence:
            return "المادة غير قابلة لإعادة التدوير."
    
    # Add other simple checks here (e.g., Fئة/Category)
    if "فئة" in question or "تصنيف" in question:
        # Example: Finds "الفئة إنشائية"
        import re
        match = re.search(r'من الفئة (\S+)\.', context_sentence)
        if match:
            return f"تصنيف المادة هو: {match.group(1)}."
            
    return None # Fallback to the regular QA model for complex questions

def is_question(text: str):
    """Return True if input looks like a question."""
    # Simple heuristic: contains question words or ends with '?'
    question_words = ["هل", "ما", "متى", "أين", "كيف", "لماذا", "كم"]
    return text.strip().endswith("؟") or any(q in text for q in question_words)

def ask_model(question: str) -> str:
    """Send Arabic question with intelligent context choice."""
    
    global last_material_discussed
    
    current_material = find_material_in_question(question)
    material_to_use = current_material if current_material else last_material_discussed

    if not material_to_use:
        return "عفواً، لم أتمكن من تحديد المادة التي تسأل عنها. يرجى ذكر مادة مثل 'خشب' أو 'خرسانة'."
    
    if not is_question(question):
        return "عفواً، لم أفهم السؤال. يرجى طرح سؤال واضح."

    simple_answer = handle_simple_yes_no(question, material_to_use)
    if simple_answer:
        # Update the tracker
        if current_material:
            last_material_discussed = current_material
        return simple_answer

    all_contexts = context_dict[material_to_use]
    num_sentences = min(3, len(all_contexts))
    random_contexts = random.sample(all_contexts, num_sentences)
    context_block = "\n".join(random_contexts) 

    qa_prompt = f"الإجابة على السؤال التالي بناءً على السياق، مع تلخيص الجوانب المختلفة إن وجدت. السؤال: {question}"

    print(f"Using context for: {material_to_use}. Current question: {question}")
  
    try:
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
        result = qa_pipeline(question=qa_prompt, context=context_block)
        answer = result["answer"]
    
        # Update the tracker only after a successful answer
        if current_material:
            last_material_discussed = current_material

    except Exception as e:
        print(f"Error during QA pipeline: {e}")
        answer = "عفواً، واجهت مشكلة تقنية أثناء معالجة السؤال التفصيلي."

    return answer.strip()

def on_speech_recognized(text: str):
    print(f" السؤال: {text}")
    pause_listening.set()
    answer = ask_model(text)
    speak(answer)
    pause_listening.clear()
    
    try:
        answer = ask_model(text)
        print(f" الإجابة: {answer}")
        done_event = threading.Event()

        #def callback():
         #   done_event.set()
        #speak(answer)
        #done_event.wait()

    except Exception as e:
        print(f" خطأ أثناء المعالجة: {e}")
        speak("حدث خطأ أثناء الإجابة على سؤالك.")
    finally:
        pause_listening.clear()

def run_recommendation_model():
    """
    Main loop for running the recommendation model.
    Handles mode selection and interaction.
    """
    speak("انتهى التحليل. يمكنك الآن البدء بطرح الأسئلة .")
    print("🎧 جاهز! يمكنك السؤال صوتياً أو كتابياً.")
    mode = None

    while True:
        if mode is None:
            mode = input("🎙️ اختر الوضع (v/t/exit): ").strip().lower()

        if mode == "exit":
            stop()
            print("تم إيقاف المساعد الصوتي.")
            break

        elif mode == "t":
            while True:
                text_question = input("⌨️ اكتب سؤالك ('back' للعودة): ").strip()
                if text_question.lower() == "back":
                    mode = None
                    break
                elif text_question.lower() == "exit":
                    stop()
                    return
                elif text_question:
                    answer = ask_model(text_question)
                    print(f"💬 الإجابة: {answer}")
                    speak(answer)

        elif mode == "v":
            print("جاري الاستماع... اطرح أسئلتك بالصوت ('exit' للخروج).")
            start_listening(on_speech_recognized)
            while True:
                time.sleep(0.5)
                # Optional: implement voice command 'exit' to break loop

        else:
            print("خيار غير معروف. اكتب 'v' أو 't' أو 'exit'.")
            mode = None


if __name__ == "__main__":
   run_recommendation_model()
