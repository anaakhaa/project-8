from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load tokenizer and model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define labels for interpretation
labels = ["1 Star (Very Negative)", "2 Stars (Negative)", "3 Stars (Neutral)", "4 Stars (Positive)", "5 Stars (Very Positive)"]

def analyze_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get the highest probability label
    sentiment = labels[torch.argmax(scores).item()]
    return sentiment, scores.tolist()

# Test with multiple languages
test_texts = {
    "English": "I absolutely love this product! It's the best thing ever!",
    "French": "Ce produit est incroyable! Je l'adore!",  # This product is amazing! I love it!
    "Spanish": "Odio este servicio. Es una pérdida de dinero.",  # I hate this service. It's a waste of money.
    "German": "Das ist eine schreckliche Erfahrung gewesen.",  # This was a terrible experience.
    "Italian": "Questo film è stato sorprendente, davvero emozionante!",  # This movie was amazing, really exciting!
}

# Run sentiment analysis on each text
for language, text in test_texts.items():
    sentiment, scores = analyze_sentiment(text)
    print(f"Language: {language}\nText: {text}\nSentiment: {sentiment}\nScores: {scores}\n{'-'*50}")
