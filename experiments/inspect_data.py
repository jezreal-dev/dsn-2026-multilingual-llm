import csv
from collections import Counter
from pathlib import Path

DATA_DIR = Path("/home/jmomoh/dsn-ai-bootcamp/data")

def audit_train():
    path = DATA_DIR / "train.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Total Rows in train.csv: {len(rows)}")
    splits = Counter(r["split"] for r in rows)
    print("Splits:", dict(splits))
    
    langs = Counter(r["lang"] for r in rows)
    print("Languages:", dict(langs))
    
    categories = Counter(r["category"] for r in rows)
    print("Categories:", dict(categories))
    
    # Check split by lang
    split_lang = Counter((r["split"], r["lang"]) for r in rows)
    print("\nSplit x Lang distribution:")
    for k, v in sorted(split_lang.items()):
        print(f"  {k}: {v}")

def audit_sample_sub():
    path = DATA_DIR / "sample_submission.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"\nTotal rows in sample_submission.csv: {len(rows)}")
    print("First 6 sample submission rows:")
    for r in rows[:6]:
        print(" ", r)

if __name__ == "__main__":
    audit_train()
    audit_sample_sub()
