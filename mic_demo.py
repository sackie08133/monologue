import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import re
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import joblib
import pyttsx3
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load pieces
clf = joblib.load("emotion_classifier.pkl")
encoder = SentenceTransformer("text_encoder")
generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct") # 1.5B params
whisper_model = whisper.load_model("base")  # ~74M params

SAMPLE_RATE = 16000
DURATION = 5  # seconds per recording

def record_audio():
    print(f"\n🎤 Recording for {DURATION} seconds... speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    wav.write("mic_input.wav", SAMPLE_RATE, audio)
    return "mic_input.wav"

def transcribe(filepath):
    result = whisper_model.transcribe(filepath, fp16=False)
    return result["text"].strip()

def predict_emotion(text):
    vec = encoder.encode([text])
    return clf.predict(vec)[0]

def trim_to_complete_sentence(text):
    """Cut off any dangling half-sentence at the end."""
    matches = list(re.finditer(r'[.!?]', text))
    if matches:
        return text[:matches[-1].end()].strip()
    return text.strip()

def dr_monologue_response(text, emotion):
    messages = [
        {"role": "system", "content": "You are Dr. Monologue, an absurdly dramatic therapist. Respond in 1-2 theatrical, overly poetic sentences most of the time. However, sometimes you will speak as if you really don't want to be with the patient. You will then go on an extremely long dialogue, of around 100 words about how the societal pressures have failed you as a comedian and you had to go to your backup job, a therapist, which in reality you deeply hate. You should speak as if you supress this feeling on a daily basis, but it leaks out occasionally. Try to be subtle. Never break character."},
        {"role": "user", "content": f"[Detected emotion: {emotion}] {text}"}
    ]
    output = generator(messages, max_new_tokens=180, do_sample=True, temperature=0.9,
                        pad_token_id=generator.tokenizer.eos_token_id)
    raw = output[0]["generated_text"][-1]["content"].strip()
    return trim_to_complete_sentence(raw)

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)

    # pick english cuz my device default is korean
    for voice in engine.getProperty("voices"):
        if "english" in voice.name.lower() or "en_" in voice.id.lower() or "en-" in voice.id.lower():
            engine.setProperty("voice", voice.id)
            break

    engine.say(text)
    engine.runAndWait()
    engine.stop()

if __name__ == "__main__":
    while True:
        input("\nPress Enter to record (or Ctrl+C to quit)...")
        filepath = record_audio()
        text = transcribe(filepath)
        print(f"You said: {text}")
        if not text:
            print("(Didn't catch anything, try again)")
            continue
        emotion = predict_emotion(text)
        response = dr_monologue_response(text, emotion)
        print(f"[Detected: {emotion}]")
        print(f"Dr. Monologue: {response}")
        speak(response)