import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load data
train_df = pd.read_csv("https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv")
dev_df = pd.read_csv("https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv")

# Turn sentences into number vectors using a small pretrained model
encoder = SentenceTransformer("all-MiniLM-L6-v2")
X_train = encoder.encode(train_df["Utterance"].tolist(), show_progress_bar=True)
y_train = train_df["Emotion"].tolist()
X_dev = encoder.encode(dev_df["Utterance"].tolist(), show_progress_bar=True)
y_dev = dev_df["Emotion"].tolist()

# Train a simple classifier on top of those vectors
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Check how well it does on unseen data
preds = clf.predict(X_dev)
print("Accuracy:", accuracy_score(y_dev, preds))

# Save everything so we can reuse it later
joblib.dump(clf, "emotion_classifier.pkl")
encoder.save("text_encoder")