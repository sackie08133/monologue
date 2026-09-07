from datasets import load_dataset

dataset = load_dataset("ajyy/MELD_audio", trust_remote_code=True)
print(dataset)
print(dataset["train"][0])