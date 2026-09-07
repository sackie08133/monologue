import time
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load our pieces (text-only classifier — it outperformed multimodal, using it by design)
clf = joblib.load("emotion_classifier.pkl")
encoder = SentenceTransformer("text_encoder")
generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

def predict_emotion(text):
    vec = encoder.encode([text])
    return clf.predict(vec)[0]

def dr_monologue_response(text, emotion):
    messages = [
        {"role": "system", "content": "You are Dr. Monologue, an absurdly dramatic therapist. Respond in 1-2 theatrical, overly poetic sentences. Never break character."},
        {"role": "user", "content": f"[Detected emotion: {emotion}] {text}"}
    ]
    output = generator(messages, max_new_tokens=60, do_sample=True, temperature=0.9,
                        pad_token_id=generator.tokenizer.eos_token_id)
    return output[0]["generated_text"][-1]["content"].strip()

def process_utterance(text, delay=2):
    """Structured output block, produced one utterance at a time."""
    emotion = predict_emotion(text)
    response = dr_monologue_response(text, emotion)
    state = {"utterance": text, "emotion": emotion}
    print(f"\n--- New input received ---")
    print(f"STATE: {state}")
    print(f"Dr. Monologue: {response}")
    time.sleep(delay)  # simulate real-time pacing between turns

if __name__ == "__main__":
    # Pull a real multi-turn dialogue from MELD to simulate a live conversation
    df = pd.read_csv("https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv")
    dialogue = df[df["Dialogue_ID"] == 5].sort_values("Utterance_ID")

    print("Streaming a real conversation, one utterance at a time...\n")
    for _, row in dialogue.iterrows():
        process_utterance(row["Utterance"])