import stanza

# List of 19 languages (ISO 639-1 codes) excluding Russian
languages = [
    "en", "es", "de", "fr", "it", "nl", "pt", "zh", "hi",
    "ar", "ja", "ko", "tr", "sv", "fi", "pl", "cs", "el", "he"
]

# Download language models
for lang in languages:
    stanza.download(lang)

# Load NLP pipelines for all languages
nlp_models = {lang: stanza.Pipeline(lang=lang, processors="tokenize,pos") for lang in languages}

# Example multilingual texts
texts = {
    "en": "Elon Musk founded SpaceX in 2002.",
    "es": "El gato negro corre rápido.",
    "de": "Angela Merkel war die Kanzlerin von Deutschland.",
    "fr": "Emmanuel Macron est le président de la France.",
    "it": "Roma è la capitale d'Italia.",
    "nl": "Amsterdam is de hoofdstad van Nederland.",
    "pt": "Cristiano Ronaldo é um jogador de futebol famoso.",
    "zh": "北京是中国的首都。",
    "hi": "महात्मा गांधी भारत के राष्ट्रपिता थे।",
    "ar": "القاهرة هي عاصمة مصر.",
    "ja": "東京は日本の首都です。",
    "ko": "서울은 대한민국의 수도입니다.",
    "tr": "İstanbul, Türkiye'nin en büyük şehridir.",
    "sv": "Stockholm är Sveriges huvudstad.",
    "fi": "Helsinki on Suomen pääkaupunki.",
    "pl": "Warszawa jest stolicą Polski.",
    "cs": "Praha je hlavní město České republiky.",
    "el": "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας.",
    "he": "ירושלים היא בירת ישראל."
}

# Function to display POS tagging results
def display_results(doc, lang):
    print(f"\n🔍 {lang.upper()} POS Tagging Results:")
    for sentence in doc.sentences:
        for word in sentence.words:
            print(f"Word: {word.text} | POS: {word.upos} | Dependency: {word.deprel}")

# Process and display results for all 19 languages
for lang, text in texts.items():
    doc = nlp_models[lang](text)
    display_results(doc, lang)
