import numpy as np
import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# --- load data subset ---
dataset = load_dataset("ajyy/MELD_audio", trust_remote_code=True)
train_data = dataset["train"].select(range(2000))  
dev_data = dataset["validation"]

# --- load pretrained audio encoder ---
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
audio_model.eval()

def get_audio_embedding(audio_array):
    inputs = audio_processor(audio_array, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = audio_model(**inputs)
    # average across time steps -> one fixed-size vector per clip
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# --- Load pretrained text encoder (reuse from before) ---
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

def build_features(data):
    text_embeds = text_encoder.encode(data["text"], show_progress_bar=True)
    audio_embeds = []
    for i, sample in enumerate(data):
        audio_embeds.append(get_audio_embedding(sample["audio"]["array"]))
        if i % 200 == 0:
            print(f"  audio {i}/{len(data)}")
    audio_embeds = np.array(audio_embeds)
    return np.concatenate([text_embeds, audio_embeds], axis=1)

print("Building training features...")
X_train = build_features(train_data)
y_train = train_data["emotion"]

print("Building validation features...")
X_dev = build_features(dev_data)
y_dev = dev_data["emotion"]

# --- Train classifier on combined text+audio features ---
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict(X_dev)
print("Multimodal accuracy:", accuracy_score(y_dev, preds))

joblib.dump(clf, "multimodal_classifier.pkl")