# Dr. Monologue: Stupid Therapist Robot
Inspired by DougDoug

Talk to it, and it'll figure out how you're feeling and answer back in character as Dr. Monologue, a dramatic therapist who occasionally lets slip that he actually wanted to be a comedian and hates this job.

## How it's built

Here's the path your voice takes:

```
You talk into the mic (5 second clip)
    │
    ▼
Whisper turns it into text (~74M params)
    │
    ▼
Text gets turned into a vector (MiniLM, ~22M) → fed into a simple classifier → emotion guess
    │
    └──► that text + emotion gets stuffed into a prompt for Dr. Monologue's personality
                │
                ▼
        Qwen2.5-1.5B writes the response (~1.5B params)
                │
                ▼
        Windows' built-in voice reads it out loud
```


| Piece | Model | Params |
|---|---|---|
| Speech-to-text | Whisper-base | ~74M |
| Turns text into a vector | all-MiniLM-L6-v2 | ~22M |
| Emotion classifier | Logistic Regression | basically nothing |
| Audio model (tried, not used in final demo) | Wav2Vec2-base | ~95M |
| Writes the response | Qwen2.5-1.5B-Instruct | ~1.5B |
| **Total** | | **~1.7B** |

None of the pretrained models were fine-tuned — they're used as-is. Only the little classifier on top was actually trained. Emotions are the standard 7 from MELD: neutral, joy, sadness, anger, fear, disgust, surprise.

## Running it yourself
Install everything first:
```
pip install torch transformers sentence-transformers scikit-learn datasets soundfile sounddevice scipy openai-whisper pyttsx3 joblib pandas
```

Then, in order:

1. `python train_text_model.py` — trains the emotion classifier on MELD text (pulls the dataset automatically)
2. `python train_multimodal_model.py` — optional, this is the audio experiment that ended up not being used in the final version
3. `python mic_demo.py` — the actual demo. Hit Enter, talk for 5 seconds, and it'll answer back out loud
4. `python stream_demo.py` — feeds in a real back-and-forth conversation from MELD instead of your mic, to show it working on a full dialogue

## Does it actually work well?

- The text-only classifier got **56% accuracy** guessing the right emotion out of 7 options (random guessing would be around 14%), which is roughly what published results on this dataset get.
- I also tried adding audio into the mix, but it actually did worse — **53%** — trained on a smaller chunk of data (2,000 samples instead of the full set)
- Takes about 5-10 seconds per response, running on a regular laptop CPU, no GPU.

## limits and darn failures
- It guesses "neutral" a lot. the training data itself is skewed that way, so the model picked up the same bias.
- Whisper sometimes mishears you, especially short phrases or anything with an accent.
- Nothing here was fine-tuned, it's all off-the-shelf pretrained models with a small classifier on top.
- The audio + text combination was pretty basic. just gluing the two feature sets together, not a real fusion setup.
- No video/vision, and no reinforcement learning, didn't get to either of those.
- It handles one full utterance at a time, not a continuous live stream — no partial responses while you're still talking.
