import joblib
from sentence_transformers import SentenceTransformer
from transformers import pipeline

clf = joblib.load("emotion_classifier.pkl")
encoder = SentenceTransformer("text_encoder")

# Small instruction-tuned model (~500M params)
generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

def predict_emotion(text):
    vec = encoder.encode([text])
    return clf.predict(vec)[0]

def dr_monologue_response(text, emotion):
    messages = [
        {"role": "system", "content": "You are Dr. Monologue, an absurdly dramatic therapist. Respond in 1-2 theatrical, overly poetic sentences. Never break character."},
        {"role": "user", "content": f"[Detected emotion: {emotion}] {text}"}
    ]
    output = generator(messages, max_new_tokens=60, do_sample=True, temperature=0.9, pad_token_id=generator.tokenizer.eos_token_id)
    return output[0]["generated_text"][-1]["content"].strip()

if __name__ == "__main__":
    while True:
        text = input("\nSay something (or 'quit'): ")
        if text.lower() == "quit":
            break
        emotion = predict_emotion(text)
        response = dr_monologue_response(text, emotion)
        print(f"[Detected: {emotion}]")
        print(f"Dr. Monologue: {response}")