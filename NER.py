import os
from transformers import pipeline

# Set backend explicitly
os.environ["TRANSFORMERS_BACKEND"] = "tensorflow"

# Load multilingual NER pipeline
ner_pipeline = pipeline(
    "ner",
    model="xlm-roberta-large-finetuned-conll03-english",
    aggregation_strategy="simple",
    framework="pt",
)


# Example multilingual text
text = (
    "Elon Musk a fondé SpaceX en 2002. "
    "OpenAI está en San Francisco. "
    "Angela Merkel war die Bundeskanzlerin von Deutschland."
)

# Run NER
ner_results = ner_pipeline(text)

# Display results
print("🔍 Cleaned Multilingual Named Entity Recognition (NER) Results:")
for entity in ner_results:
    print(f"Entity: {entity['word']} | Type: {entity['entity_group']} | Score: {entity['score']:.4f}")
