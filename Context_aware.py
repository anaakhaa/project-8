import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoModelForTokenClassification
from langdetect import detect
from deep_translator import GoogleTranslator
import stanza
import pandas as pd
from datetime import datetime
import os
import time

# ==== Model Metadata ====
MODEL_VERSION = "1.0"
EXECUTION_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ==== Load Pretrained Models ====
sentiment_model = "nlptown/bert-base-multilingual-uncased-sentiment"
emotion_model = "joeddav/distilbert-base-uncased-go-emotions-student"
irony_model = "cardiffnlp/twitter-roberta-base-irony"
ner_model = "xlm-roberta-large-finetuned-conll03-english"
harassment_model = "Hate-speech-CNERG/dehatebert-mono-english"

sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model)

emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model)

irony_tokenizer = AutoTokenizer.from_pretrained(irony_model)
irony_model = AutoModelForSequenceClassification.from_pretrained(irony_model)

ner_tokenizer = AutoTokenizer.from_pretrained(ner_model)
ner_model_pt = AutoModelForTokenClassification.from_pretrained(ner_model)

harassment_tokenizer = AutoTokenizer.from_pretrained(harassment_model)
harassment_model = AutoModelForSequenceClassification.from_pretrained(harassment_model)

ner_pipeline = pipeline("ner", model=ner_model_pt, tokenizer=ner_tokenizer, aggregation_strategy="simple", framework="pt")
languages = ["en", "es", "de", "fr", "it", "nl", "pt", "zh", "hi", "ar", "ja", "ko", "tr", "sv", "fi", "pl", "cs", "el", "he"]
nlp_models = {lang: stanza.Pipeline(lang=lang, processors="tokenize,pos") for lang in languages}

sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
emotion_labels = emotion_model.config.id2label


def simulate_thinking():
    print("\n\U0001F9E0 Analyzing...", end="", flush=True)
    for _ in range(3):
        time.sleep(0.4)
        print(".", end="", flush=True)
    print(" ✅\n")

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_text(text, target="en"):
    lang = detect_language(text)
    if lang != "en":
        return GoogleTranslator(source='auto', target=target).translate(text)
    return text

def analyze_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = sentiment_labels[torch.argmax(scores).item()]
    return sentiment, scores.tolist()

def predict_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    emotion = emotion_labels.get(predicted_class, "Unknown")
    return emotion

def detect_irony(text):
    inputs = irony_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = irony_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    irony_label = torch.argmax(probs, dim=1).item()
    return "Ironic" if irony_label == 1 else "Not Ironic"

def extract_entities(text):
    return ner_pipeline(text)

def pos_tagging(text, lang="en"):
    if lang in nlp_models:
        doc = nlp_models[lang](text)
        return [(word.text, word.upos) for sentence in doc.sentences for word in sentence.words]
    return []

def detect_harassment(text):
    inputs = harassment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = harassment_model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
    labels = ["normal", "offensive", "hate"]
    predicted_class = labels[torch.argmax(scores).item()]
    return predicted_class

def compute_chs(sentiment, emotion, irony, entities, harassment_class, verbose=False):
    score = 0.0
    weights = {
        "sentiment": 0.25,
        "emotion": 0.15,
        "irony": 0.2,
        "entities": 0.2,
        "harassment": 0.2
    }

    if sentiment in ["Very Negative", "Negative"]:
        score += weights["sentiment"]
        if verbose:
            print(f"🟥 Sentiment '{sentiment}' is negative → +{weights['sentiment']}")

    high_risk_emotions = ["anger", "fear", "disgust", "sadness"]
    if emotion.lower() in high_risk_emotions:
        score += weights["emotion"]
        if verbose:
            print(f"🔥 High-risk emotion '{emotion}' detected → +{weights['emotion']}")

    if irony == "Ironic":
        score += weights["irony"]
        if verbose:
            print(f"🌀 Irony detected → +{weights['irony']}")

    if any(ent['entity_group'] in ["PER", "ORG", "LOC"] for ent in entities):
        score += weights["entities"]
        if verbose:
            print(f"🏷️ Sensitive named entity detected → +{weights['entities']}")

    if harassment_class in ["offensive", "hate"]:
        score += weights["harassment"]
        if verbose:
            print(f"🚫 Harassment detected ({harassment_class}) → +{weights['harassment']}")

    normalized_score = round(score / sum(weights.values()), 2)
    if verbose:
        print(f"📊 Final CHS Score: {normalized_score} / 1.0")

    return normalized_score


def classify_risk(chs_score):
    if chs_score >= 0.6:
        return "🔴 High Risk"
    elif chs_score >= 0.3:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

def save_to_excel(result, filename="analysis_log.xlsx"):
    flat_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_text": result["original_text"],
        "translated_text": result["translated_text"],
        "sentiment": result["sentiment"],
        "emotion": result["emotion"],
        "emotion_type": result.get("emotion_type", "Neutral"),
        "irony": result["irony"],
        "harassment": result["harassment"],
        "chs": result["chs"],
        "risk_tier": result["risk_tier"],
        "entities": ', '.join([ent["word"] for ent in result["entities"]]),
        "pos_tags": ', '.join([f"{word}/{tag}" for word, tag in result["pos_tags"]])
    }
    df = pd.DataFrame([flat_result])

    if os.path.exists(filename):
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            startrow = pd.read_excel(filename).shape[0] + 1
            df.to_excel(writer, index=False, header=False, startrow=startrow)
    else:
        df.to_excel(filename, index=False, engine='openpyxl')


def unified_analysis(text):
    simulate_thinking()
    translated_text = translate_text(text)
    sentiment, _ = analyze_sentiment(translated_text)
    emotion = predict_emotion(translated_text)
    irony = detect_irony(translated_text)
    harassment_class = detect_harassment(translated_text)  # <- this gives the class: "normal", "offensive", "hate"
    entities = extract_entities(translated_text)
    pos_tags = pos_tagging(translated_text)

    # Now pass all five things into compute_chs
    chs = compute_chs(sentiment, emotion, irony, entities, harassment_class)
    risk_tier = classify_risk(chs)

    emotion_type = "Negative" if sentiment in ["Very Negative", "Negative"] else (
        "Positive" if sentiment in ["Positive", "Very Positive"] else "Neutral")

    return {
        "original_text": text,
        "translated_text": translated_text,
        "sentiment": sentiment,
        "emotion": emotion,
        "emotion_type": emotion_type,
        "irony": irony,
        "harassment": harassment_class,  # <- store the harassment_class here, not just harassment flag
        "entities": entities,
        "pos_tags": pos_tags,
        "chs": chs,
        "risk_tier": risk_tier
    }


def display_analysis_result(results):
    print("\n==============================")
    print("\U0001F4CA ANALYSIS RESULT")
    print("------------------------------")
    print(f"\U0001F4CC Original Text     : {results['original_text']}")
    print(f"\U0001F310 Translated Text   : {results['translated_text']}")
    print(f"\U0001F4AC Sentiment         : {results['sentiment']}")
    print(f"\U0001F3AD Emotion           : {results['emotion']}")
    print(f"\U0001F9ED Emotion Type      : {results.get('emotion_type', 'N/A')}")
    print(f"\U0001F300 Irony             : {results['irony']}")
    print(f"🚫 Harassment         : {results['harassment']}")
    print(f"\U0001F3F7️ Named Entities    : {', '.join([ent['word'] for ent in results['entities']]) if results['entities'] else '[]'}")
    print(f"\U0001F524 POS Tags          : {', '.join([f'{word}/{tag}' for word, tag in results['pos_tags']])}")
    print(f"\U0001F4C8 CHS Score         : {results['chs']}")
    print(f"\U0001F6A8 Threat Level       : {results['risk_tier']}")
    print("==============================")
    print(f"\U0001F9EA Model Version: {MODEL_VERSION} | ⏱️ Time: {EXECUTION_TIME}")
    print("==============================")
