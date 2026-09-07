import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import re
import numpy as np
import torch
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import joblib
import pyttsx3
from sentence_transformers import SentenceTransformer
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2Model

# --- Load all pieces once, at startup ---
clf = joblib.load("multimodal_classifier.pkl")          # trained on text+audio combined
text_encoder = SentenceTransformer("text_encoder")
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
audio_model.eval()
generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
whisper_model = whisper.load_model("base")  # ~74M params

SAMPLE_RATE = 16000
DURATION = 5  # seconds per recording

def record_audio():
    print(f"\n🎤 Recording for {DURATION} seconds... speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    wav.write("mic_input.wav", SAMPLE_RATE, audio)
    audio_array = audio.flatten()  # same audio, as a flat 1D array for Wav2Vec
    return "mic_input.wav", audio_array

def transcribe(filepath):
    result = whisper_model.transcribe(filepath, fp16=False)
    return result["text"].strip()

def get_audio_embedding(audio_array):
    inputs = audio_processor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        outputs = audio_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def predict_emotion(text, audio_array):
    text_vec = text_encoder.encode([text])[0]
    audio_vec = get_audio_embedding(audio_array)
    combined = np.concatenate([text_vec, audio_vec]).reshape(1, -1)
    return clf.predict(combined)[0]

def trim_to_complete_sentence(text):
    # cut off any dangling half-sentence at the end
    matches = list(re.finditer(r'[.!?]', text))
    if matches:
        return text[:matches[-1].end()].strip()
    return text.strip()

def dr_monologue_response(text, emotion):
    messages = [
        {"role": "system", "content": "You are Dr. Monologue, an absurdly dramatic therapist. Respond in 1-2 theatrical, overly poetic sentences. Never break character."},
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
        filepath, audio_array = record_audio()
        text = transcribe(filepath)
        print(f"You said: {text}")
        if not text:
            print("(Didn't catch anything, try again)")
            continue
        emotion = predict_emotion(text, audio_array)
        response = dr_monologue_response(text, emotion)
        print(f"[Detected: {emotion}]")
        print(f"Dr. Monologue: {response}")
        speak(response)