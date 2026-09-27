import pytest
import pandas as pd
import re

# 1. Constants under test
VALID_LABELS = ["business", "health", "politics", "religion", "sports", "entertainment", "technology"]
LANG_NAME_MAP = {
    "hau": "Hausa",
    "ibo": "Igbo",
    "pcm": "Nigerian Pidgin",
    "yor": "Yoruba"
}

LANGUAGE_CATEGORY_MAP = {
    "hau": ["business", "entertainment", "health", "politics", "religion", "sports", "technology"],
    "ibo": ["business", "entertainment", "health", "politics", "religion", "sports"],
    "pcm": ["business", "entertainment", "health", "politics", "sports"],
    "yor": ["entertainment", "health", "politics", "religion", "sports"]
}

# 2. Functions under test
def build_topic_prompt(text: str, language: str) -> str:
    lang_full = LANG_NAME_MAP.get(language, language)
    valid_cats = LANGUAGE_CATEGORY_MAP.get(language, VALID_LABELS)
    cats_str = ", ".join(valid_cats)
    lead_text = text[:600]
    return (
        f"<|im_start|>system\nYou are an expert multilingual news editor specializing in African languages.<|im_end|>\n"
        f"<|im_start|>user\nClassify the following {lang_full} news article into exactly one of these categories: {cats_str}.\n\n"
        f"Article:\n{lead_text}\n\n"
        f"Respond with only the category name in lowercase.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def build_headline_prompt(text: str, language: str, topic: str) -> str:
    lang_full = LANG_NAME_MAP.get(language, language)
    parts = text.split(".")
    lead = ".".join(parts[:2]).strip()[:400]
    if not lead:
        lead = text[:300].strip()
    remaining = ".".join(parts[2:]).strip()[:400]
    context_str = f"\n\nContext:\n{remaining}" if remaining else ""
    return (
        f"<|im_start|>system\nYou are an expert news editor in {lang_full}.<|im_end|>\n"
        f"<|im_start|>user\nWrite a complete, punchy, and accurate news headline in {lang_full} for this {topic} article based on the lead event.\n\n"
        f"Lead:\n{lead}{context_str}\n\n"
        f"Headline:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def parse_topic_output(raw_text: str, language: str = None) -> str:
    cleaned = raw_text.strip().lower()
    valid_cats = LANGUAGE_CATEGORY_MAP.get(language, VALID_LABELS) if language else VALID_LABELS
    for label in valid_cats:
        if label in cleaned:
            return label
    return "politics" if "politics" in valid_cats else valid_cats[0]


# --- Test Cases ---

def test_language_category_map_integrity():
    """Verify that all categories mapped to languages are valid and match MasakhaNEWS specs."""
    for lang, cats in LANGUAGE_CATEGORY_MAP.items():
        assert set(cats).issubset(set(VALID_LABELS)), f"Invalid categories for {lang}: {set(cats) - set(VALID_LABELS)}"
    
    assert len(LANGUAGE_CATEGORY_MAP["hau"]) == 7
    assert len(LANGUAGE_CATEGORY_MAP["ibo"]) == 6
    assert len(LANGUAGE_CATEGORY_MAP["pcm"]) == 5
    assert len(LANGUAGE_CATEGORY_MAP["yor"]) == 5

    # Specific missing category assertions
    assert "technology" not in LANGUAGE_CATEGORY_MAP["ibo"]
    assert "technology" not in LANGUAGE_CATEGORY_MAP["pcm"]
    assert "technology" not in LANGUAGE_CATEGORY_MAP["yor"]
    assert "religion" not in LANGUAGE_CATEGORY_MAP["pcm"]
    assert "business" not in LANGUAGE_CATEGORY_MAP["yor"]


def test_build_topic_prompt_yoruba_restriction():
    """Verify Yoruba topic prompt does not leak business or technology."""
    article = "Aare Bola Tinubu ti fi owo si iwe ofin titun lati ran awon odo lowo."
    prompt = build_topic_prompt(article, "yor")
    
    assert "Yoruba" in prompt
    assert "business" not in prompt
    assert "technology" not in prompt
    assert "entertainment, health, politics, religion, sports" in prompt
    assert article in prompt


def test_build_topic_prompt_pidgin_restriction():
    """Verify Pidgin topic prompt does not leak religion or technology."""
    article = "Di Nigeria Labour Congress don declare strike because of petrol price."
    prompt = build_topic_prompt(article, "pcm")
    
    assert "Nigerian Pidgin" in prompt
    assert "religion" not in prompt
    assert "technology" not in prompt
    assert "business, entertainment, health, politics, sports" in prompt


def test_parse_topic_output_leakage_prevention():
    """Verify parse_topic_output prevents predicting disallowed categories for a language."""
    # If the raw output hallucinated "business" for Yoruba:
    parsed_yor = parse_topic_output("This is a business article", language="yor")
    assert parsed_yor != "business", "Yoruba must never predict business"
    assert parsed_yor in LANGUAGE_CATEGORY_MAP["yor"]

    # For Hausa, "business" should be accepted
    parsed_hau = parse_topic_output("This is a business article", language="hau")
    assert parsed_hau == "business"

    # If the raw output hallucinated "religion" for Pidgin:
    parsed_pcm = parse_topic_output("The pastor said religion is good", language="pcm")
    assert parsed_pcm != "religion", "Pidgin must never predict religion"
    assert parsed_pcm in LANGUAGE_CATEGORY_MAP["pcm"]


def test_build_headline_prompt_inverted_pyramid():
    """Verify headline prompt isolates Lead sentence and includes Context."""
    text = "First sentence here. Second sentence here. Third sentence with background context. Fourth sentence."
    prompt = build_headline_prompt(text, "yor", "politics")

    assert "Lead:\nFirst sentence here. Second sentence here" in prompt
    assert "Context:\nThird sentence with background context. Fourth sentence" in prompt
    assert "complete, punchy, and accurate news headline in Yoruba" in prompt
    assert "politics" in prompt


def test_build_headline_prompt_single_sentence():
    """Verify headline prompt gracefully handles single sentence articles without empty context crashes."""
    text = "Just one short sentence without follow-up."
    prompt = build_headline_prompt(text, "hau", "sports")

    assert "Lead:\nJust one short sentence without follow-up" in prompt
    assert "complete, punchy, and accurate news headline in Hausa" in prompt


def test_rebalancing_exact_instance_count():
    """Verify training dataset rebalancing creates exactly 8,800 rows and 17,600 samples."""
    # Simulate train distribution
    data = []
    for lang, count in [("hau", 2219), ("yor", 1433), ("ibo", 1356), ("pcm", 1060)]:
        for i in range(count):
            data.append({
                "id": f"{lang}_{i}",
                "language": lang,
                "text": f"Sample text {i} for {lang}.",
                "label": "politics",
                "headline": f"Sample headline {i}"
            })
    train_df = pd.DataFrame(data)

    target_lang_count = 2200
    balanced_dfs = []
    for lang, group in train_df.groupby("language"):
        if len(group) < target_lang_count:
            oversampled = group.sample(target_lang_count, replace=True, random_state=42)
            balanced_dfs.append(oversampled)
        else:
            balanced_dfs.append(group.sample(target_lang_count, random_state=42))

    balanced_train_df = pd.concat(balanced_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    assert len(balanced_train_df) == 8800
    for lang in ["hau", "ibo", "pcm", "yor"]:
        assert (balanced_train_df["language"] == lang).sum() == 2200

    # Multi-task compilation
    train_samples = []
    for _, row in balanced_train_df.iterrows():
        train_samples.append({"prompt": "A", "target": "A"})
        train_samples.append({"prompt": "B", "target": "B"})

    assert len(train_samples) == 17600
