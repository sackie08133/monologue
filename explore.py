import pandas as pd

url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv"
df = pd.read_csv(url)

print(df.head())
print(df["Emotion"].unique())