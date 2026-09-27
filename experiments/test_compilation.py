import pandas as pd
import numpy as np

# Load local data
train_df = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/data/train.csv')
dev_df = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/data/dev.csv')
test_df = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/data/test.csv')

# Canonical column standardization
CANONICAL_COLS = ["id", "language", "text", "label", "headline"]
for df in [train_df, dev_df]:
    if "category" in df.columns and "label" not in df.columns:
        df["label"] = df["category"]
    if "lang" in df.columns and "language" not in df.columns:
        df["language"] = df["lang"]

if "lang" in test_df.columns and "language" not in test_df.columns:
    test_df["language"] = test_df["lang"]

train_df = train_df[CANONICAL_COLS].copy()
dev_df = dev_df[CANONICAL_COLS].copy()

LANGUAGE_CATEGORY_MAP = {
    "hau": ["business", "entertainment", "health", "politics", "religion", "sports", "technology"],
    "ibo": ["business", "entertainment", "health", "politics", "religion", "sports"],
    "pcm": ["business", "entertainment", "health", "politics", "sports"],
    "yor": ["entertainment", "health", "politics", "religion", "sports"]
}

LANG_NAME_MAP = {
    "hau": "Hausa",
    "ibo": "Igbo",
    "pcm": "Nigerian Pidgin",
    "yor": "Yoruba"
}
VALID_LABELS = ["business", "health", "politics", "religion", "sports", "entertainment", "technology"]

def rebalance_stratified(df, target_per_slice=280):
    """
    ML Best Practice: Stratified dual-key rebalancing over (language, category).
    Oversamples underrepresented minority classes to eliminate severe Macro-F1 degradation.
    """
    balanced_dfs = []
    for lang, valid_cats in LANGUAGE_CATEGORY_MAP.items():
        for cat in valid_cats:
            sub = df[(df["language"] == lang) & (df["label"] == cat)]
            if len(sub) == 0:
                continue
            if len(sub) < target_per_slice:
                n_needed = target_per_slice - len(sub)
                resampled = sub.sample(n=n_needed, replace=True, random_state=42)
                balanced_dfs.append(pd.concat([sub, resampled]))
            else:
                balanced_dfs.append(sub)
    result = pd.concat(balanced_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return result

balanced_train_df = rebalance_stratified(train_df, target_per_slice=280)
print(f"Original Train: {train_df.shape} -> Balanced Train: {balanced_train_df.shape}")

# Prompt builders
def build_topic_prompt(text: str, language: str) -> str:
    lang_full = LANG_NAME_MAP.get(language, language)
    valid_cats = LANGUAGE_CATEGORY_MAP.get(language, VALID_LABELS)
    cats_str = ", ".join(valid_cats)
    truncated_text = text[:800]
    return (
        f"<|im_start|>system\nYou are an expert multilingual news editor specializing in African languages.<|im_end|>\n"
        f"<|im_start|>user\nClassify the following {lang_full} news article into exactly one of these categories: {cats_str}.\n\n"
        f"Article:\n{truncated_text}\n\n"
        f"Respond with only the category name in lowercase.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def build_topic_prompt_v2(text: str, language: str) -> str:
    lang_full = LANG_NAME_MAP.get(language, language)
    valid_cats = LANGUAGE_CATEGORY_MAP.get(language, VALID_LABELS)
    cats_str = ", ".join(valid_cats)
    truncated_text = text[:800]
    return (
        f"<|im_start|>system\nYou are an expert {lang_full} news classification specialist.<|im_end|>\n"
        f"<|im_start|>user\nIdentify the primary topic category for this {lang_full} news report. Options: {cats_str}.\n\n"
        f"Article text:\n{truncated_text}\n\n"
        f"Topic:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def build_headline_prompt(text: str, language: str, topic: str) -> str:
    lang_full = LANG_NAME_MAP.get(language, language)
    truncated_text = text[:800]
    return (
        f"<|im_start|>system\nYou are an expert news editor in {lang_full}.<|im_end|>\n"
        f"<|im_start|>user\nWrite a concise, accurate headline in {lang_full} for this {topic} article.\n\n"
        f"Article:\n{truncated_text}\n\n"
        f"Headline:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# Compile instruction samples with balanced task weighting
train_samples = []
for _, row in balanced_train_df.iterrows():
    # Task A: Primary Topic Classification Prompt
    prompt_a1 = build_topic_prompt(row["text"], row["language"])
    target_a1 = f"{row['label']}<|im_end|>"
    train_samples.append({"prompt": prompt_a1, "target": target_a1, "task": "topic"})

    # Task A (Variant 2): Invariant Topic Classification Prompt (Task balancing)
    prompt_a2 = build_topic_prompt_v2(row["text"], row["language"])
    target_a2 = f"{row['label']}<|im_end|>"
    train_samples.append({"prompt": prompt_a2, "target": target_a2, "task": "topic"})

    # Task B: Headline Generation (conditioned on ground truth topic)
    prompt_b = build_headline_prompt(row["text"], row["language"], row["label"])
    target_b = f"{row['headline']}<|im_end|>"
    train_samples.append({"prompt": prompt_b, "target": target_b, "task": "headline"})

print(f"Total compiled multi-task samples: {len(train_samples)}")
task_counts = pd.Series([s["task"] for s in train_samples]).value_counts()
print("Task Distribution:")
print(task_counts)
