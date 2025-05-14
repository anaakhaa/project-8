from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load the HateXplain fine-tuned model (example model - you can choose any from HF)
model_name = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Labels mapping based on HateXplain (hate, offensive, normal)
labels = {0: "normal", 1: "offensive", 2: "hate"}

def detect_sexual_harassment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]  # probabilities

    pred_label = torch.argmax(probs).item()
    confidence = round(float(probs[pred_label]), 3)
    return {
        "predicted_class": labels[pred_label],
        "confidence": confidence,
        "raw_scores": {labels[i]: round(float(probs[i]), 3) for i in range(len(labels))}
    }

# Test sentence
if __name__ == "__main__":
    test_sentence = "She deserved that because she dressed like that."
    result = detect_sexual_harassment(test_sentence)
    print("\n--- Harassment Detection Result ---")
    print(f"Text: {test_sentence}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Raw Scores: {result['raw_scores']}")
