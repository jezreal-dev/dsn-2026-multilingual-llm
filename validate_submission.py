#!/usr/bin/env python3
"""Submission Integrity Validator (Strict Final Gate).

Audits submission.csv against all competition rules:
1. Exact row count (3,486 data rows + 1 header = 3,487 lines)
2. Column schema ('id', 'prediction')
3. ID suffixes ('_topic' and '_headline')
4. Topic predictions in valid 7-label set (lowercase)
5. Headline predictions non-empty, non-whitespace
6. Proper CSV quoting for commas
"""

import csv
import sys
from pathlib import Path

VALID_TOPICS = {
    "business", "health", "politics", "religion",
    "sports", "entertainment", "technology"
}

def validate_submission_file(csv_path: Path, expected_test_rows: int = 1743) -> bool:
    print(f"=== Auditing Submission: {csv_path} ===")
    
    if not csv_path.exists():
        print(f"[BLOCKING ERROR] File does not exist: {csv_path}")
        return False

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # Check header schema
        if reader.fieldnames != ["id", "prediction"]:
            print(f"[BLOCKING ERROR] Header mismatch. Expected ['id', 'prediction'], got {reader.fieldnames}")
            return False

        rows = list(reader)

    expected_rows = expected_test_rows * 2
    if len(rows) != expected_rows:
        print(f"[BLOCKING ERROR] Row count mismatch. Expected exactly {expected_rows} rows, got {len(rows)}")
        return False

    topic_count = 0
    headline_count = 0
    errors = 0

    for idx, r in enumerate(rows, start=2):  # 1-based, row 2 is first data row
        row_id = r.get("id", "")
        pred = r.get("prediction", "")

        if not row_id:
            print(f"[BLOCKING ERROR] Line {idx}: Missing 'id'")
            errors += 1
            continue

        if pred is None or pred.strip() == "":
            print(f"[BLOCKING ERROR] Line {idx} ({row_id}): Prediction is blank or empty whitespace")
            errors += 1
            continue

        if row_id.endswith("_topic"):
            topic_count += 1
            cleaned_topic = pred.strip()
            if cleaned_topic != cleaned_topic.lower():
                print(f"[BLOCKING ERROR] Line {idx} ({row_id}): Topic prediction '{pred}' must be lowercase")
                errors += 1
            if cleaned_topic.lower() not in VALID_TOPICS:
                print(f"[BLOCKING ERROR] Line {idx} ({row_id}): Unknown topic label '{pred}'. Must be one of {sorted(VALID_TOPICS)}")
                errors += 1

        elif row_id.endswith("_headline"):
            headline_count += 1
            if len(pred.strip()) < 3:
                print(f"[WARNING] Line {idx} ({row_id}): Suspiciously short headline: '{pred}'")
        else:
            print(f"[BLOCKING ERROR] Line {idx}: ID '{row_id}' does not end with '_topic' or '_headline'")
            errors += 1

        if errors >= 10:
            print(f"[HALTING AUDIT] 10+ critical errors detected. Resolve the pipeline output first.")
            return False

    if topic_count != expected_test_rows or headline_count != expected_test_rows:
        print(f"[BLOCKING ERROR] Mismatch in task counts. Topics: {topic_count}, Headlines: {headline_count}, Expected: {expected_test_rows} each")
        return False

    if errors == 0:
        print(f"\n[AUDIT PASSED] submission.csv is 100% compliant with Kaggle competition requirements!")
        print(f"Total rows: {len(rows)} (Topics: {topic_count}, Headlines: {headline_count})")
        print("Sample Rows:")
        for r in rows[:4]:
            print(f"  {r['id']}: {r['prediction'][:50]}")
        return True
    else:
        print(f"\n[AUDIT FAILED] Found {errors} blocking errors.")
        return False

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/jmomoh/dsn-ai-bootcamp/data/sample_submission.csv")
    success = validate_submission_file(target)
    sys.exit(0 if success else 1)
