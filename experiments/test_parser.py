import re

LANGUAGE_CATEGORY_MAP = {
    "hau": ["business", "entertainment", "health", "politics", "religion", "sports", "technology"],
    "ibo": ["business", "entertainment", "health", "politics", "religion", "sports"],
    "pcm": ["business", "entertainment", "health", "politics", "sports"],
    "yor": ["entertainment", "health", "politics", "religion", "sports"]
}
VALID_LABELS = ["business", "health", "politics", "religion", "sports", "entertainment", "technology"]

def parse_topic_output(raw_text: str, language: str = None) -> str:
    valid_cats = LANGUAGE_CATEGORY_MAP.get(language, VALID_LABELS) if language else VALID_LABELS
    words = re.findall(r'\b[a-zA-Z]+\b', raw_text.lower())
    for w in words:
        if w in valid_cats:
            return w
    cleaned = raw_text.lower()
    for cat in valid_cats:
        if cat in cleaned:
            return cat
    fallbacks = {
        "hau": "politics",
        "ibo": "politics",
        "pcm": "sports",
        "yor": "politics"
    }
    return fallbacks.get(language, "politics" if "politics" in valid_cats else valid_cats[0])

# Test edge cases
cases = [
    ("health", "hau", "health"),
    ("Health<|im_end|>", "ibo", "health"),
    ("The category is sports and not business.", "yor", "sports"), # In old code, business matched first!
    ("Technology", "yor", "politics"), # technology invalid in Yoruba -> falls back to politics!
    ("business news", "pcm", "business"),
    ("unknown output", "pcm", "sports"),
    ("religion and faith", "hau", "religion"),
]

for raw, lang, expected in cases:
    res = parse_topic_output(raw, lang)
    assert res == expected, f"Failed on {raw} for {lang}: got {res}, expected {expected}"
    print(f"PASS: raw='{raw}', lang='{lang}' => parsed='{res}'")

print("\nAll parser test cases passed successfully!")
