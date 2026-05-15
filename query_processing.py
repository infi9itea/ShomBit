import re
from typing import List

BANGLISH_MAP: dict[str, str] = {
    # Admission / application
    "vorti": "ভর্তি", "bhorti": "ভর্তি", "bhorti": "ভর্তি",
    "admission": "ভর্তি", "apply": "আবেদন", "application": "আবেদন",
    "deadline": "শেষ তারিখ", "last date": "শেষ তারিখ",

    # Fees / finance
    "fee": "ফি", "fees": "ফি", "taka": "টাকা",
    "tuition": "টিউশন ফি", "tuition fee": "টিউশন ফি",
    "credit": "ক্রেডিট", "waiver": "ওয়েভার",
    "scholarship": "বৃত্তি", "scholarship info": "বৃত্তির তথ্য",

    # People
    "sir": "স্যার", "madam": "ম্যাডাম", "teacher": "শিক্ষক",
    "faculty": "শিক্ষকমণ্ডলী", "professor": "অধ্যাপক",
    "head": "বিভাগীয় প্রধান", "chairman": "চেয়ারম্যান",

    # Academic
    "cgpa": "CGPA", "gpa": "GPA", "grade": "গ্রেড",
    "result": "ফলাফল", "exam": "পরীক্ষা", "semester": "সেমিস্টার",
    "course": "কোর্স", "subject": "বিষয়", "credit hour": "ক্রেডিট আওয়ার",
    "retake": "রিটেক", "retake exam": "রিটেক পরীক্ষা",
    "probation": "প্রবেশন", "dismissal": "বহিষ্কার",
    "class": "ক্লাস", "routine": "রুটিন", "schedule": "সময়সূচি",
    "syllabus": "সিলেবাস", "department": "বিভাগ",

    # University life
    "library": "লাইব্রেরি", "canteen": "ক্যান্টিন",
    "bus": "বাস", "transport": "পরিবহন",
    "hostel": "হোস্টেল", "dormitory": "ডরমিটরি",
    "club": "ক্লাব", "event": "অনুষ্ঠান",

    # Question words
    "koto": "কত", "kobe": "কবে", "ki": "কি", "kothay": "কোথায়",
    "keno": "কেন", "kivabe": "কীভাবে", "boro": "বড়",
    "choto": "ছোট", "ache": "আছে", "nai": "নেই",
    "bolun": "বলুন", "janun": "জানুন", "jante chai": "জানতে চাই",
    "somporke": "সম্পর্কে", "jonno": "জন্য",

    # Common EWU-specific
    "ewu": "ইস্ট ওয়েস্ট ইউনিভার্সিটি",
    "east west": "ইস্ট ওয়েস্ট ইউনিভার্সিটি",
    "cse": "কম্পিউটার বিজ্ঞান ও প্রকৌশল",
    "eee": "তড়িৎ ও ইলেকট্রনিক প্রকৌশল",
    "bba": "ব্যবসায় প্রশাসন",
    "mba": "মাস্টার্স অব বিজনেস অ্যাডমিনিস্ট্রেশন",
    "llb": "আইন স্নাতক",
    "pharmacy": "ফার্মেসি",
    "civil": "পুরকৌশল",
}

_CHAR_REPLACEMENTS = {
    "bh": "b",
    "ph": "f",
    "kh": "k",
    "gh": "g",
    "sh": "s",
    "th": "t",
}


def _collapse_repeated_chars(text: str) -> str:
    """hellooo → hello"""
    return re.sub(r"(.)\1{2,}", r"\1", text)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = _collapse_repeated_chars(text)
    for old, new in _CHAR_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text


def transliterate_query(text: str) -> str:
    """Replace Banglish tokens with Bangla script equivalents (longest-match first)."""
    result = text
    for phrase in sorted(BANGLISH_MAP.keys(), key=len, reverse=True):
        if phrase in result:
            result = result.replace(phrase, BANGLISH_MAP[phrase])
    return result


def expand_query(query: str) -> List[str]:

    normalized     = normalize_text(query)
    transliterated = transliterate_query(normalized)

    variants = [query, normalized, transliterated]
    seen, unique = set(), []
    for v in variants:
        if v not in seen and v.strip():
            seen.add(v)
            unique.append(v)
    return unique
