import re

BANGLISH_MAP = {
    "vorti": "ভর্তি",
    "bhorti": "ভর্তি",
    "admission": "ভর্তি",
    "koto": "কত",
    "kobe": "কবে",
    "fee": "ফি",
    "taka": "টাকা",
    "sir": "স্যার",
    "cgpa": "CGPA",
}

def normalize_text(text):
    text = text.lower().strip()

    text = re.sub(r'(.)\1{2,}', r'\1', text)

    replacements = {
        "bh": "b",
        "ph": "f",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text

def transliterate_query(text):
    words = text.split()

    converted = []
    for word in words:
        converted.append(BANGLISH_MAP.get(word, word))

    return " ".join(converted)

def expand_query(query):
    normalized = normalize_text(query)
    transliterated = transliterate_query(normalized)

    expanded = list(set([
        query,
        normalized,
        transliterated
    ]))

    return expanded