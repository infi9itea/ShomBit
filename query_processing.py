from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────
# Lexicon:  romanized_key -> (bangla_script, english_canonical)
# One entry per concept. Spelling variants are handled by _fold(), so you do
# NOT need to list every romanisation by hand.
# ──────────────────────────────────────────────────────────────────
LEXICON: Dict[str, Tuple[str, str]] = {
    # Admission / application
    "vorti": ("ভর্তি", "admission"),
    "bhorti": ("ভর্তি", "admission"),
    "admission": ("ভর্তি", "admission"),
    "apply": ("আবেদন", "apply"),
    "application": ("আবেদন", "application"),
    "deadline": ("শেষ তারিখ", "deadline"),
    "last date": ("শেষ তারিখ", "last date"),
    "last day": ("শেষ দিন", "last day"),
    "admitted": ("ভর্তিকৃত", "admitted"),
    "waitlist": ("অপেক্ষমাণ তালিকা", "waitlist"),
    "merit list": ("মেধা তালিকা", "merit list"),
    "registration": ("নিবন্ধন", "registration"),
    "enroll": ("তালিকাভুক্ত", "enroll"),

    # Fees / finance
    "fee": ("ফি", "fee"),
    "fees": ("ফি", "fees"),
    "taka": ("টাকা", "taka"),
    "tk": ("টাকা", "taka"),
    "tuition": ("টিউশন ফি", "tuition"),
    "tuition fee": ("টিউশন ফি", "tuition fee"),
    "credit": ("ক্রেডিট", "credit"),
    "waiver": ("ওয়েভার", "waiver"),
    "fine": ("জরিমানা", "fine"),
    "scholarship": ("বৃত্তি", "scholarship"),
    "stipend": ("বৃত্তি", "stipend"),
    "payment": ("পেমেন্ট", "payment"),
    "installment": ("কিস্তি", "installment"),
    "due": ("বকেয়া", "due"),
    "financial aid": ("আর্থিক সহায়তা", "financial aid"),
    "refund": ("অর্থ ফেরত", "refund"),

    # People
    "sir": ("স্যার", "sir"),
    "madam": ("ম্যাডাম", "madam"),
    "teacher": ("শিক্ষক", "teacher"),
    "faculty": ("শিক্ষকমণ্ডলী", "faculty"),
    "professor": ("অধ্যাপক", "professor"),
    "head": ("বিভাগীয় প্রধান", "department head"),
    "chairman": ("চেয়ারম্যান", "chairman"),
    "dean": ("ডীন", "dean"),
    "advisor": ("উপদেষ্টা", "advisor"),
    "student": ("শিক্ষার্থী", "student"),
    "alumni": ("প্রাক্তন শিক্ষার্থী", "alumni"),

    # Academic
    "cgpa": ("সিজিপিএ", "cgpa"),
    "gpa": ("জিপিএ", "gpa"),
    "grade": ("গ্রেড", "grade"),
    "marks": ("নম্বর", "marks"),
    "result": ("ফলাফল", "result"),
    "exam": ("পরীক্ষা", "exam"),
    "semester": ("সেমিস্টার", "semester"),
    "course": ("কোর্স", "course"),
    "subject": ("বিষয়", "subject"),
    "credit hour": ("ক্রেডিট আওয়ার", "credit hour"),
    "retake": ("রিটেক", "retake"),
    "makeup exam": ("মেকআপ পরীক্ষা", "makeup exam"),
    "probation": ("প্রবেশন", "probation"),
    "routine": ("রুটিন", "class routine"),
    "schedule": ("সময়সূচি", "schedule"),
    "syllabus": ("সিলেবাস", "syllabus"),
    "department": ("বিভাগ", "department"),
    "attendance": ("উপস্থিতি", "attendance"),
    "midterm": ("মধ্যমেয়াদী", "midterm"),
    "final": ("চূড়ান্ত", "final exam"),
    "quiz": ("কুইজ", "quiz"),
    "assignment": ("অ্যাসাইনমেন্ট", "assignment"),
    "thesis": ("থিসিস", "thesis"),
    "grading": ("গ্রেডিং", "grading"),

    # University life
    "library": ("লাইব্রেরি", "library"),
    "canteen": ("ক্যান্টিন", "canteen"),
    "bus": ("বাস", "bus"),
    "transport": ("পরিবহন", "transport"),
    "club": ("ক্লাব", "club"),
    "event": ("অনুষ্ঠান", "event"),
    "gym": ("জিম", "gym"),
    "sports": ("খেলাধুলা", "sports"),
    "orientation": ("ওরিয়েন্টেশন", "orientation"),
    "convocation": ("সমাবর্তন", "convocation"),
    "campus": ("ক্যাম্পাস", "campus"),
    "lab": ("ল্যাব", "lab"),
    "wifi": ("ওয়াইফাই", "wifi"),

    # Question / function words (Banglish -> meaning)
    "koto": ("কত", "how much"),
    "kobe": ("কবে", "when"),
    "kothay": ("কোথায়", "where"),
    "keno": ("কেন", "why"),
    "nai": ("নেই", "none"),
    "jonno": ("জন্য", "for"),
    "somporke": ("সম্পর্কে", "about"),
    "jante chai": ("জানতে চাই", "want to know"),
    "ki": ("কী", "what"),

    # Dates / time
    "today": ("আজ", "today"),
    "tomorrow": ("আগামীকাল", "tomorrow"),
    "semester fee": ("সেমিস্টার ফি", "semester fee"),

    # Departments  (English canonical kept verbatim so it matches English data)
    "cse": ("কম্পিউটার বিজ্ঞান ও প্রকৌশল", "computer science and engineering"),
    "computer science": ("কম্পিউটার বিজ্ঞান", "computer science"),
    "eee": ("তড়িৎ ও ইলেকট্রনিক প্রকৌশল", "electrical and electronic engineering"),
    "ece": ("ইলেকট্রনিক্স ও যোগাযোগ প্রকৌশল", "electronics and communications engineering"),
    "bba": ("ব্যবসায় প্রশাসন", "business administration"),
    "mba": ("এমবিএ", "master of business administration"),
    "llb": ("আইন স্নাতক", "bachelor of laws"),
    "law": ("আইন", "law"),
    "pharmacy": ("ফার্মেসি", "pharmacy"),
    "civil": ("পুরকৌশল", "civil engineering"),
    "english": ("ইংরেজি বিভাগ", "english department"),
    "economics": ("অর্থনীতি", "economics"),
    "sociology": ("সমাজবিজ্ঞান", "sociology"),

    # Programs
    "undergraduate": ("স্নাতক", "undergraduate"),
    "postgraduate": ("স্নাতকোত্তর", "postgraduate"),
    "bachelor": ("স্নাতক", "bachelor"),
    "masters": ("স্নাতকোত্তর", "masters"),

    # EWU specific
    "ewu": ("ইস্ট ওয়েস্ট ইউনিভার্সিটি", "east west university"),
    "ewubd": ("ইস্ট ওয়েস্ট ইউনিভার্সিটি", "east west university"),

    # Utilities
    "contact": ("যোগাযোগ", "contact"),
    "email": ("ইমেইল", "email"),
    "phone": ("ফোন", "phone"),
    "address": ("ঠিকানা", "address"),
    "location": ("অবস্থান", "location"),

    # Time / deadline terms missing from lexicon
    "shomoyshima":  ("সময়সীমা",    "deadline"),
    "shesh shomoy": ("শেষ সময়",   "last date"),
    "shesh din":    ("শেষ দিন",    "last day"),
    "abedoner":     ("আবেদনের",    "application"),
    "abedon":       ("আবেদন",      "application"),
    "bortoman":     ("বর্তমান",    "current"),
    "shomoy":       ("সময়",        "time"),
    "shima":        ("সীমা",       "limit"),
    "kivabe":       ("কীভাবে",     "how to"),
    "shuvidha":     ("সুবিধা",     "benefit"),
    "sujog":        ("সুযোগ",      "opportunity"),
    "programgular": ("প্রোগ্রামগুলোর", "programs"),
    "kono":         ("কোনো",       "any"),
    "ache":         ("আছে",        "available"),
    "jante":        ("জানতে",      "to know"),
}

# Two-char folds applied before single-char folds. These are intentionally
# "lossy" — their only job is to make spelling variants hash to the same key.
_FOLDS: List[Tuple[str, str]] = [
    ("bh", "b"), ("ph", "f"), ("kh", "k"), ("gh", "g"), ("sh", "s"),
    ("th", "t"), ("ch", "c"), ("jh", "j"), ("dh", "d"), ("rh", "r"),
    ("oo", "u"), ("ee", "i"), ("ii", "i"), ("uu", "u"), ("aa", "a"),
    ("v", "b"), ("w", "b"), ("z", "j"), ("y", "i"), ("q", "k"),
]

_BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u0980-\u09FF]+")
_MAX_PHRASE = 3  # longest multi-token lexicon entry (e.g. "jante chai")


def has_bangla(text: str) -> bool:
    """True if the string contains any Bangla-script character."""
    return bool(_BANGLA_RE.search(text))


def _fold(token: str) -> str:
    """Deterministic fuzzy key. Symmetric across lexicon keys and query tokens."""
    t = token.lower()
    for a, b in _FOLDS:
        t = t.replace(a, b)
    t = re.sub(r"(.)\1+", r"\1", t)              # collapse any repeated run -> single
    t = re.sub(r"[^a-z0-9\u0980-\u09FF]", "", t)
    return t


def _fold_phrase(tokens: List[str]) -> str:
    return " ".join(_fold(t) for t in tokens if _fold(t))


# Pre-fold every lexicon key once at import time.
_FOLD_LEXICON: Dict[str, Tuple[str, str]] = {}
for _k, _v in LEXICON.items():
    _FOLD_LEXICON[_fold_phrase(_k.split())] = _v


def normalize_text(text: str) -> str:
    """Light surface cleanup only (NOT the destructive fold)."""
    t = text.lower().strip()
    t = re.sub(r"(.)\1{2,}", r"\1", t)   # hellooo -> hello (runs of 3+)
    t = re.sub(r"\s+", " ", t)
    return t


def _transliterate(query: str) -> Tuple[str, str]:
    """Token/phrase-aware mapping. Returns (english_variant, bangla_variant)."""
    toks = _TOKEN_RE.findall(query.lower())
    en: List[str] = []
    bn: List[str] = []
    i = 0
    while i < len(toks):
        matched = False
        for n in range(_MAX_PHRASE, 0, -1):        # longest-match first
            if i + n <= len(toks):
                key = _fold_phrase(toks[i:i + n])
                if key and key in _FOLD_LEXICON:
                    bn_val, en_val = _FOLD_LEXICON[key]
                    en.append(en_val)
                    bn.append(bn_val)
                    i += n
                    matched = True
                    break
        if not matched:                            # unknown token -> keep as-is
            en.append(toks[i])
            bn.append(toks[i])
            i += 1
    return " ".join(en), " ".join(bn)


def expand_query(query: str) -> List[str]:
    """
    Expand a (possibly Banglish) query into de-duplicated variants for hybrid
    retrieval: the raw query, a normalised form, an English-canonical form, and
    a Bangla-script form. The retriever runs dense + sparse search over each and
    fuses with RRF, so coverage spans English and Bangla content simultaneously.
    """
    base = query.strip()
    norm = normalize_text(base)
    en_variant, bn_variant = _transliterate(base)

    variants = [base, norm, en_variant, bn_variant]
    seen, unique = set(), []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


if __name__ == "__main__":
    # Quick self-test demonstrating the bugs that are now fixed.
    for q in [
        "CSE vorti fee koto?",
        "bhorti deadline kobe?",
        "What is the tuition fee for CSE?",
        "কম্পিউটার বিজ্ঞান বিভাগে ভর্তির যোগ্যতা কি?",
        "EEE er faculty members ke ke?",
    ]:
        print(f"\nQ: {q}")
        for v in expand_query(q):
            print("   ->", v)