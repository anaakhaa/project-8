import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langdetect import detect
from deep_translator import GoogleTranslator

# Optional: Define a clean cache directory to avoid corrupted downloads
#HF_CACHE_DIR = "./hf_temp_cache"
#os.makedirs(HF_CACHE_DIR, exist_ok=True)

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "joeddav/distilbert-base-uncased-go-emotions-student"

# Updated tokenizer and model loading
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)


# Get the label mapping (ID to Emotion)
label_map = model.config.id2label


# Function to detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


# Function for emotion prediction with language support
def predict_emotion(text):
    lang = detect_language(text)

    # Translate if not English
    if lang != "en":
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        print(f"🔄 Translated ({lang} ➝ en): {translated_text}")
    else:
        translated_text = text

    # Tokenization & Prediction
    inputs = tokenizer(translated_text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()

    # Get the emotion label & confidence
    emotion = label_map[predicted_class]
    confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()

    print(f"Input: {text}")
    print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2f})\n")
    return emotion


# Test Cases in Multiple Languages
test_sentences = [
    "I am so happy today!",  # English
    "Me siento muy triste y solo.",  # Spanish
    "今日はとても楽しいです！",  # Japanese
    "Ich bin wütend auf dich!",  # German
    "Je suis nerveux pour l'examen.",  # French
    "那個笑話真的很好笑！",  # Chinese
]

# Run predictions
for sentence in test_sentences:
    predict_emotion(sentence)

# Example usage
user_input = input("Enter a sentence to analyze emotion: ")
predict_emotion(user_input)
