import csv
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("/home/jmomoh/dsn-ai-bootcamp/data")
DEV_PATH = DATA_DIR / "dev.csv"

# Language configs mapping
LANG_CONFIGS = {
    "hau": "hau",
    "ibo": "ibo",
    "pcm": "pcm",
    "yor": "yor"
}

# The 7 valid competition categories
VALID_CATEGORIES = {
    "business", "health", "politics", "religion",
    "sports", "entertainment", "technology"
}

def build_dev_dataset() -> None:
    print("Loading official MasakhaNEWS validation splits...")
    dev_rows = []
    
    for lang, config in LANG_CONFIGS.items():
        ds = load_dataset("masakhane/masakhanews", config, split="validation")
        print(f"Loaded {lang} validation: {len(ds)} rows")
        
        for idx, item in enumerate(ds):
            category = item.get("label", item.get("category", "")).lower().strip()
            headline = item.get("headline", "").strip()
            text = item.get("text", "").strip()
            url = item.get("url", "")
            # ID convention: lang_xxxx
            row_id = f"{lang}_dev_{idx+1:04d}"
            
            assert category in VALID_CATEGORIES, f"Unknown category '{category}' in {lang}"
            assert len(text) > 0, f"Empty text for row {row_id}"
            assert len(headline) > 0, f"Empty headline for row {row_id}"
            
            dev_rows.append({
                "category": category,
                "headline": headline,
                "text": text,
                "url": url,
                "id": row_id,
                "split": "dev",
                "lang": lang
            })
            
    print(f"\nTotal validation rows compiled: {len(dev_rows)}")
    assert len(dev_rows) == 869, f"Expected 869 dev rows, got {len(dev_rows)}"
    
    # Write to dev.csv
    fieldnames = ["category", "headline", "text", "url", "id", "split", "lang"]
    with open(DEV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dev_rows)
        
    print(f"Successfully generated audit-ready dev set: {DEV_PATH}")

if __name__ == "__main__":
    build_dev_dataset()
