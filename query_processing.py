import re
from typing import List

BANGLISH_MAP: dict[str, str] = {
    # Admission / application
    "vorti": "ভর্তি", "bhorti": "ভর্তি", "vorthi": "ভর্তি",
    "admission": "ভর্তি", "apply": "আবেদন", "application": "আবেদন",
    "deadline": "শেষ তারিখ", "last date": "শেষ তারিখ", "last day": "শেষ দিন",
    "admitted": "ভর্তিকৃত", "rejected": "বাতিল", "waitlist": "অপেক্ষমাণ তালিকা",
    "merit list": "মেধা তালিকা", "roll number": "রোল নাম্বার", "registration": "নিবন্ধন",
    "enroll": "তালিকাভুক্ত",

    # Fees / finance
    "fee": "ফি", "fees": "ফি", "taka": "টাকা", "tk": "টাকা",
    "tuition": "টিউশন ফি", "tuition fee": "টিউশন ফি",
    "credit": "ক্রেডিট", "waiver": "ওয়েভার", "fine": "জরিমানা",
    "scholarship": "বৃত্তি", "scholarship info": "বৃত্তির তথ্য",
    "stipend": "বৃত্তি", "grant": "অনুদান", "loan": "ঋণ",
    "payment": "পেমেন্ট", "installment": "কিস্তি", "due": "বকেয়া",
    "financial aid": "আর্থিক সহায়তা", "discount": "ছাড়", "refund": "অর্থ ফেরত",

    # People
    "sir": "স্যার", "madam": "ম্যাডাম", "teacher": "শিক্ষক", "instructor": "প্রশিক্ষক",
    "faculty": "শিক্ষকমণ্ডলী", "professor": "অধ্যাপক", "dr": "ডাক্তার",
    "head": "বিভাগীয় প্রধান", "chairman": "চেয়ারম্যান", "dean": "ডীন",
    "advisor": "উপদেষ্টা", "mentor": "পরামর্শদাতা", "counselor": "পরামর্শদাতা",
    "student": "শিক্ষার্থী", "alumni": "প্রাক্তন শিক্ষার্থী", "parent": "অভিভাবক",

    # Academic
    "cgpa": "CGPA", "gpa": "GPA", "grade": "গ্রেড", "mark": "নম্বর", "marks": "নম্বর",
    "result": "ফলাফল", "exam": "পরীক্ষা", "test": "পরীক্ষা", "assessment": "মূল্যায়ন",
    "semester": "সেমিস্টার", "term": "টার্ম", "quarter": "ত্রৈমাসিক",
    "course": "কোর্স", "subject": "বিষয়", "credit hour": "ক্রেডিট আওয়ার",
    "retake": "রিটেক", "retake exam": "রিটেক পরীক্ষা", "makeup exam": "মেকআপ পরীক্ষা",
    "probation": "প্রবেশন", "dismissal": "বহিষ্কার", "suspension": "স্থগন",
    "class": "ক্লাস", "routine": "রুটিন", "schedule": "সময়সূচি", "timetable": "সময়সূচী",
    "syllabus": "সিলেবাস", "department": "বিভাগ", "faculty": "অনুষদ",
    "attendance": "উপস্থিতি", "absent": "অনুপস্থিত", "present": "উপস্থিত",
    "midterm": "মধ্যমেয়াদী", "final": "চূড়ান্ত", "quiz": "ক্যুইজ",
    "assignment": "অ্যাসাইনমেন্ট", "project": "প্রকল্প", "thesis": "থিসিস",
    "gpc": "GPC", "standing": "অবস্থান", "academic standing": "একাডেমিক অবস্থান",

    # University life
    "library": "লাইব্রেরি", "canteen": "ক্যান্টিন", "cafeteria": "ক্যাফেটেরিয়া",
    "bus": "বাস", "transport": "পরিবহন", "shuttle": "শাটেল",
    "hostel": "হোস্টেল", "dormitory": "ডরমিটরি", "dorm": "ডরম",
    "club": "ক্লাব", "event": "অনুষ্ঠান", "activity": "কার্যক্রম",
    "gym": "জিম", "sports": "খেলাধুলা", "athletics": "অ্যাথলেটিক্স",
    "orientation": "প্রবেশপর্বণী", "convocation": "সমাবর্তন", "commencement": "স্নাতক অনুষ্ঠান",
    "campus": "ক্যাম্পাস", "office": "অফিস", "lab": "ল্যাব", "laboratory": "ল্যাবরেটরি",
    "parking": "পার্কিং", "wifi": "ওয়াইফাই", "internet": "ইন্টারনেট",

    # Question words
    "koto": "কত", "kobe": "কবে", "ki": "কি", "kothay": "কোথায়", "kotha": "কোথা",
    "keno": "কেন", "kivabe": "কীভাবে", "boro": "বড়", "ka": "কা",
    "choto": "ছোট", "ache": "আছে", "nai": "নেই", "asa": "আসা",
    "bolun": "বলুন", "janun": "জানুন", "jante chai": "জানতে চাই",
    "somporke": "সম্পর্কে", "jonno": "জন্য", "konojon": "কোনো জন",
    "when": "কখন", "where": "কোথায়", "why": "কেন", "how": "কীভাবে", "what": "কী",
    "which": "কোনটা", "who": "কে",

    # Dates and time
    "today": "আজ", "tomorrow": "আগামীকাল", "yesterday": "গতকাল",
    "week": "সপ্তাহ", "month": "মাস", "year": "বছর", "day": "দিন",
    "morning": "সকাল", "afternoon": "বিকেল", "evening": "সন্ধ্যা", "night": "রাত",
    "time": "সময়", "date": "তারিখ",

    # Departments
    "cse": "কম্পিউটার বিজ্ঞান ও প্রকৌশল", "computer science": "কম্পিউটার বিজ্ঞান",
    "eee": "তড়িৎ ও ইলেকট্রনিক প্রকৌশল", "electrical": "তড়িৎ",
    "bba": "ব্যবসায় প্রশাসন", "business": "ব্যবসা",
    "mba": "মাস্টার্স অব বিজনেস অ্যাডমিনিস্ট্রেশন",
    "llb": "আইন স্নাতক", "law": "আইন",
    "pharmacy": "ফার্মেসি", "pharma": "ফার্মা",
    "civil": "পুরকৌশল", "civil engineering": "নাগরিক প্রকৌশল",
    "arch": "স্থাপত্য", "architecture": "স্থাপত্য",
    "english": "ইংরেজি বিভাগ", "eng": "ইংরেজি",
    "bangla": "বাংলা বিভাগ", "bn": "বাংলা",

    # Common EWU-specific
    "ewu": "ইস্ট ওয়েস্ট ইউনিভার্সিটি",
    "east west": "ইস্ট ওয়েস্ট ইউনিভার্সিটি",
    "ewubd": "ইস্ট ওয়েস্ট ইউনিভার্সিটি বাংলাদেশ",
    "uc": "আপটাউন ক্যাম্পাস", "uptown": "আপটাউন",

    # Courses and programs
    "undergraduate": "স্নাতক", "postgraduate": "স্নাতকোত্তর",
    "bachelor": "স্নাতক", "masters": "স্নাতকোত্তর", "diploma": "ডিপ্লোমা",
    "honours": "সম্মান", "major": "বিশেষায়ন", "minor": "গৌণ",

    # General utilities
    "error": "ত্রুটি", "problem": "সমস্যা", "issue": "সমস্যা", "help": "সাহায্য",
    "contact": "যোগাযোগ", "call": "কল", "email": "ইমেইল", "phone": "ফোন",
    "address": "ঠিকানা", "location": "অবস্থান", "map": "মানচিত্র",
}

_CHAR_REPLACEMENTS = {
    "bh": "b",
    "ph": "f",
    "kh": "k",
    "gh": "g",
    "sh": "s",
    "th": "t",
    "ch": "c",
    "jh": "j",
    "dh": "d",
    "nh": "n",
    "rh": "r",
    "lh": "l",
    "wh": "w",
    "ng": "ng",
    "aa": "a",
    "oo": "o",
    "ee": "e",
    "ii": "i",
    "uu": "u",
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
