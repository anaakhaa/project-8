from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the pretrained irony detection model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-irony"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def detect_irony(text):
    """
    Detects irony in the given text using a pretrained RoBERTa model.

    Parameters:
    - text (str): The input text to analyze.

    Returns:
    - str: "Ironic" if irony is detected, otherwise "Not Ironic".
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Compute probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    irony_label = torch.argmax(probs, dim=1).item()

    return "Ironic" if irony_label == 1 else "Not Ironic"


# Example usage
if __name__ == "__main__":
    test_sentences = [
        "Oh great, another Monday! Just what I needed.",  # Ironic
        "I love spending hours in traffic every day!",  # Ironic
        "I had a really good day today!",  # Not Ironic
        "The weather is perfect for a picnic."  # Not Ironic
    ]

    for sentence in test_sentences:
        print(f"Text: {sentence}\nIrony Detection: {detect_irony(sentence)}\n")
