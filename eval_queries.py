# Hardcoded evaluation set — 60 queries across 20 intents × 3 languages.
# Imported by eval_all.py so the notebook stays as thin as the ingest notebook.

_RS = {
    1:  ["https://ewubd.edu/undergraduate-dates-deadline",
         "https://ewubd.edu/graduate-dates-deadline"],
    2:  ["EWU Admission JSON"],
    3:  ["EWU Admission JSON", "EWU Admission Requirements JSON"],
    4:  ["EWU Academic Structure JSON"],
    5:  ["EWU Academic Structure JSON", "EWU Faculty JSON"],
    6:  ["EWU Tuition Fees JSON", "EWU Programs JSON"],
    7:  ["EWU Institutional Information JSON"],
    8:  ["EWU Tuition Fees JSON"],
    9:  ["EWU Scholarships JSON"],
    10: ["EWU Scholarships JSON"],
    11: ["https://ewubd.edu/undergraduate-tuition-fees", "EWU Tuition Fees JSON"],
    12: ["EWU Institutional Information JSON"],
    13: ["EWU Institutional Information JSON"],
    14: ["EWU Campus Life Facilities JSON", "EWU Facilities JSON"],
    15: ["EWU Helpdesk JSON"],
    16: ["EWU Helpdesk JSON"],
    17: ["https://ewubd.edu/events"],
    18: ["EWU Admission Requirements JSON"],
    19: ["EWU Campus Life Facilities JSON"],
    20: ["East West University Student Conduct and Discipline Policy", "EWU Programs JSON"],
}

# (seq, language, question, ground_truth)
_RAW = [
    # ── English (1–20) ──────────────────────────────────────────────────────
    (1,  "English", "What is the current application deadline for undergraduate programs?",
     "April 15, 2026."),
    (2,  "English", "How can I apply for admission?",
     "Apply through EWU's online admission portal. Create a new applicant profile, "
     "choose the program, pay the application fee (cash at the Admission Office or "
     "online via bKash), then sign in with the EWU Login ID and mobile number to "
     "complete and submit the online form."),
    (3,  "English", "What documents are needed for the admission process?",
     "A recent passport-size colored photo (JPG, max 100 KB) and a scanned signature "
     "(JPG, max 60 KB). All academic certificates and mark sheets/transcripts in "
     "original and photocopy form; The originals will be returned after verification."),
    (4,  "English", "How many courses are there in the CSE department?",
     "The current official EWU CSE page does not present one single fixed total number "
     "of courses on the public page because the total varies with elective choices."),
    (5,  "English", "Can you show me the list of departments offered by the university?",
     "Computer Science & Engineering, Electrical and Electronic Engineering, Genetic "
     "Engineering & Biotechnology, Pharmacy, Civil Engineering, Mathematical & Physical "
     "Sciences, English, Law, Social Relations, Information Studies, Sociology, "
     "Business Administration, Economics"),
    (6,  "English", "Tell me about the BBA total credits.",
     "The BBA program requires 123 total credits."),
    (7,  "English", "Why should I get admitted into EWU?",
     "Top-quality faculty, labs and research/library support, free full-time medical "
     "services, and quality education at affordable cost. The university also highlights "
     "a large library collection and a clean, spacious, air-conditioned campus."),
    (8,  "English", "What is the current tuition fee for Computer Science?",
     "Tuition fee is Tk 6,500 per credit and grand total Tk 1,003,400."),
    (9,  "English", "Are there any scholarship opportunities?", "Yes"),
    (10, "English", "What are the terms and conditions for merit scholarships?",
     "100% tuition-free for 4 years for candidates with A+ in all subjects in the most "
     "recent SSC & HSC, or 7 As in O-level + 3 As in A-level. 50% tuition waiver for "
     "1 year for candidates with GPA 5.00 in the most recent SSC & HSC, subject to "
     "qualifying in the EWU admission test, maintaining at least GPA 3.50 each semester "
     "as a regular student, and following EWU disciplinary/code-of-conduct rules."),
    (11, "English", "Has there been a recent change in the per credit fee for CSE courses?",
     "Recent change not conclusively verified from the live official pages."),
    (12, "English", "Where is East West University located?",
     "A/2, Jahurul Islam Avenue, Jahurul Islam City, Aftabnagar, Dhaka-1212, Bangladesh"),
    (13, "English", "Can you tell me about some notable EWU alumni?",
     "Md. Abir Hasan, Md. Nazmul Khan, Fatima Tasnim, Redoan Rony, MD Fahim, "
     "Md. Shafayat Hossain, Miraj Ahmed, Mohammad Sajjad Islam Shejan, Rubana Huq, Afran Nisho"),
    (14, "English", "Is there any medical center in the university?", "Yes"),
    (15, "English", "How can I contact the registrar's office?", "registrar@ewubd.edu"),
    (16, "English", "Give me the email of the CSE department helpdesk.", "helpdesk-cse@ewubd.edu"),
    (17, "English", "Are there any events happening this month?", "Spring in the Air 2026"),
    (18, "English", "Can international students apply for admission?", "Yes"),
    (19, "English",
     "Does the university offer on-campus or affiliated dormitory/hostel accommodation?", "No"),
    (20, "English", "Can I transfer my credits?", "Yes"),
    # ── Bangla (21–40) ──────────────────────────────────────────────────────
    (21, "Bangla", "স্নাতক প্রোগ্রামগুলোর জন্য আবেদনের বর্তমান শেষ তারিখ কবে?",
     "১৫ই এপ্রিল, ২০২৬"),
    (22, "Bangla", "আমি কীভাবে ভর্তির জন্য আবেদন করতে পারি?",
     "অনলাইন ভর্তি পোর্টালের মাধ্যমে আবেদন করুন। ভর্তি সাইটে যান, একটি নতুন আবেদনকারী "
     "প্রোফাইল তৈরি করুন, প্রোগ্রাম নির্বাচন করুন, ১৫০০ টাকা আবেদন ফি প্রদান করুন, তারপর "
     "ইডব্লিউইউ লগইন আইডি এবং মোবাইল নম্বর দিয়ে সাইন ইন করে অনলাইন ফর্মটি পূরণ ও জমা দিন।"),
    (23, "Bangla", "ভর্তি প্রক্রিয়ার জন্য কী কী কাগজপত্র প্রয়োজন?",
     "সাম্প্রতিক পাসপোর্ট আকারের একটি রঙিন ছবি (JPG, সর্বোচ্চ ১০০ KB) এবং একটি স্ক্যান করা "
     "স্বাক্ষর (JPG, সর্বোচ্চ ৬০ KB)। সকল শিক্ষাগত যোগ্যতার সনদপত্র এবং মার্কশিটের মূল ও ফটোকপি।"),
    (24, "Bangla", "সিএসই বিভাগে কয়টি কোর্স আছে?",
     "The current official EWU CSE page does not present one single fixed total number "
     "of courses because the total varies with elective choices."),
    (25, "Bangla", "বিশ্ববিদ্যালয়ে যে বিভাগগুলো রয়েছে সেগুলোর তালিকা দেখাতে পারবে?",
     "কম্পিউটার সায়েন্স অ্যান্ড ইঞ্জিনিয়ারিং, ইলেকট্রিক্যাল অ্যান্ড ইলেকট্রনিক ইঞ্জিনিয়ারিং, "
     "জেনেটিক ইঞ্জিনিয়ারিং অ্যান্ড বায়োটেকনোলজি, ফার্মেসি, সিভিল ইঞ্জিনিয়ারিং, ম্যাথেমেটিক্যাল "
     "অ্যান্ড ফিজিক্যাল সায়েন্সেস, ইংরেজি, আইন, সমাজ সম্পর্ক, ইনফরমেশন স্টাডিজ, সমাজবিজ্ঞান, "
     "ব্যবসা প্রশাসন, অর্থনীতি"),
    (26, "Bangla", "বিবিএ কোর্সের মোট ক্রেডিট সম্পর্কে বলো।",
     "বিবিএ প্রোগ্রামের জন্য মোট ১২৩ ক্রেডিট প্রয়োজন।"),
    (27, "Bangla", "আমি কেন ইস্ট ওয়েস্ট ইউনিভার্সিটিতে ভর্তি হব?",
     "শীর্ষ মানের শিক্ষকবৃন্দ, অত্যাধুনিক গবেষণাগার এবং গবেষণা ও গ্রন্থাগার সহায়তা, "
     "চিকিৎসা সেবা এবং সাশ্রয়ী মূল্যে মানসম্মত শিক্ষা।"),
    (28, "Bangla", "কম্পিউটার সায়েন্সের বর্তমান টিউশন ফি কত?",
     "টিউশন ফি প্রতি ক্রেডিটে ৬,৫০০ টাকা এবং সর্বমোট ফি ১০,০৩,৪০০ টাকা।"),
    (29, "Bangla", "কোনো স্কলারশিপের সুযোগ আছে কি?", "হ্যাঁ"),
    (30, "Bangla", "মেরিট স্কলারশিপের শর্তাবলি কী?",
     "সর্বশেষ এসএসসি ও এইচএসসি-তে সকল বিষয়ে এ+ প্রাপ্ত প্রার্থীদের জন্য ৪ বছরের জন্য ১০০% "
     "টিউশন-ফ্রি। সর্বশেষ এসএসসি ও এইচএসসি-তে জিপিএ ৫.০০ প্রাপ্ত প্রার্থীদের জন্য ১ বছরের "
     "জন্য ৫০% টিউশন ফি মওকুফ।"),
    (31, "Bangla", "সিএসই কোর্সগুলোর প্রতি ক্রেডিট ফিতে কি সম্প্রতি কোনো পরিবর্তন হয়েছে?",
     "অফিসিয়াল পেজ থেকে চূড়ান্তভাবে যাচাই করা হয়নি।"),
    (32, "Bangla", "ইস্ট ওয়েস্ট ইউনিভার্সিটি কোথায় অবস্থিত?",
     "এ/২, জহুরুল ইসলাম এভিনিউ, জহুরুল ইসলাম সিটি, আফতাবনগর, ঢাকা-১২১২, বাংলাদেশ"),
    (33, "Bangla", "EWU-এর কিছু উল্লেখযোগ্য প্রাক্তন শিক্ষার্থীর সম্পর্কে বলতে পারবে?",
     "এম. ডি. আবির হাসান, এম. ডি. নাজমুল খান, ফাতিমা তাসনিম, রেদোয়ান রনি, এমডি ফাহিম, "
     "এম. ডি. শাফায়াত হোসেন, মিরাজ আহমেদ, মোহাম্মদ সাজ্জাদ ইসলাম শেজান, রুবানা হক, আফরান নিশো"),
    (34, "Bangla", "বিশ্ববিদ্যালয়ে কি কোনো চিকিৎসা কেন্দ্র আছে?", "হ্যাঁ"),
    (35, "Bangla", "আমি কীভাবে রেজিস্ট্রারের অফিসের সাথে যোগাযোগ করতে পারি?",
     "registrar@ewubd.edu"),
    (36, "Bangla", "আমাকে সিএসই বিভাগের হেল্পডেস্কের ইমেইলটি দিন।", "helpdesk-cse@ewubd.edu"),
    (37, "Bangla", "এই মাসে কি কোনো অনুষ্ঠান হচ্ছে?", "স্প্রিং ইন দ্য এয়ার ২০২৬"),
    (38, "Bangla", "আন্তর্জাতিক শিক্ষার্থীরা কি ভর্তির জন্য আবেদন করতে পারে?", "হ্যাঁ"),
    (39, "Bangla",
     "বিশ্ববিদ্যালয় কি ক্যাম্পাসের ভেতরে বা অনুমোদিত ছাত্রাবাস/হোস্টেলে থাকার ব্যবস্থা করে?",
     "না"),
    (40, "Bangla", "আমি কি আমার ক্রেডিটগুলো স্থানান্তর করতে পারি?", "হ্যাঁ"),
    # ── Banglish (41–60) ────────────────────────────────────────────────────
    (41, "Banglish",
     "Undergraduate programgular bortoman vorti abedoner shesh shomoyshima ki?",
     "April 15, 2026."),
    (42, "Banglish", "Ami kibhabe vorti-r jonno abedon korte pari?",
     "Apply through EWU's online admission portal. Create a new applicant profile, "
     "choose the program, pay the application fee, then sign in to complete and submit the form."),
    (43, "Banglish", "Vorti prokriyar jonno ki ki document proyojon?",
     "A recent passport-size colored photo (JPG, max 100 KB) and a scanned signature "
     "(JPG, max 60 KB). All academic certificates in original and photocopy form."),
    (44, "Banglish", "CSE bibhage koyti course ache?",
     "The current official EWU CSE page does not present one single fixed total number "
     "of courses because the total varies with elective choices."),
    (45, "Banglish", "Bishwabidyaloye je bibhaggulo royeche segulor talika dekhate parbe?",
     "Computer Science & Engineering, Electrical and Electronic Engineering, Genetic "
     "Engineering & Biotechnology, Pharmacy, Civil Engineering, Mathematical & Physical "
     "Sciences, English, Law, Social Relations, Information Studies, Sociology, "
     "Business Administration, Economics"),
    (46, "Banglish", "BBA er mot credit somporke bolo.",
     "The BBA program requires 123 total credits."),
    (47, "Banglish", "Ami keno East West University-te vorti hobo?",
     "Top-quality faculty, labs and research/library support, free full-time medical "
     "services, and quality education at affordable cost."),
    (48, "Banglish", "Computer Science-er bortoman tuition fee koto?",
     "Tuition fee is Tk 6,500 per credit and grand total Tk 1,003,400."),
    (49, "Banglish", "Kono scholarship-er sujog ache ki?", "Yes"),
    (50, "Banglish", "Merit scholarship-er shortaboli ki?",
     "100% tuition-free for 4 years for candidates with A+ in all subjects in the most "
     "recent SSC & HSC, or 7 As in O-level + 3 As in A-level. 50% tuition waiver for "
     "1 year for candidates with GPA 5.00."),
    (51, "Banglish", "CSE coursegular proti credit fee-te ki shomproti kono poriborton hoyeche?",
     "Recent change not conclusively verified from the live official pages."),
    (52, "Banglish", "East West University kothay obosthito?",
     "A/2, Jahurul Islam Avenue, Jahurul Islam City, Aftabnagar, Dhaka-1212, Bangladesh"),
    (53, "Banglish", "EWU-er kichu ullokkhoggo praktan shikkharthir somporke bolte parbe?",
     "Md. Abir Hasan, Md. Nazmul Khan, Fatima Tasnim, Redoan Rony, MD Fahim, "
     "Md. Shafayat Hossain, Miraj Ahmed, Mohammad Sajjad Islam Shejan, Rubana Huq, Afran Nisho"),
    (54, "Banglish", "Bishwabidyaloye ki kono medical center ache?", "Yes"),
    (55, "Banglish", "Ami kibhabe registrar office-er sathe jogajog korte pari?",
     "registrar@ewubd.edu"),
    (56, "Banglish", "CSE department-er helpdesk-er email dao.", "helpdesk-cse@ewubd.edu"),
    (57, "Banglish", "Ei mashe ki kono event hocche?", "Spring in the Air 2026"),
    (58, "Banglish", "International shikkharthira ki vorti-r jonno abedon korte pare?", "Yes"),
    (59, "Banglish", "Bishwabidyaloy ki on-campus ba shongjukto dormitory/hostel subidha dey?",
     "No"),
    (60, "Banglish", "Ami ki amar credit transfer korte pari?", "Yes"),
]

QUERIES = [
    {
        "seq":              seq,
        "language":         lang,
        "question":         question,
        "ground_truth":     gt,
        "relevant_sources": _RS[((seq - 1) % 20) + 1],
    }
    for seq, lang, question, gt in _RAW
]
