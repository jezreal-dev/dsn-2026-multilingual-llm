import pandas as pd

train_df = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/data/train.csv')
dev_df = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/data/dev.csv')
test_df = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/data/test.csv')

print("=== CHECKING FOR MISSING / NULL VALUES (ML Best Practice) ===")
print("Train nulls:")
print(train_df.isnull().sum())
print("\nDev nulls:")
print(dev_df.isnull().sum())
print("\nTest nulls:")
print(test_df.isnull().sum())

print("\n=== SAMPLE HEADLINES BY LANGUAGE ===")
for lang in ['hau', 'ibo', 'pcm', 'yor']:
    sub = train_df[train_df['lang'] == lang].head(3)
    print(f"\n--- {lang.upper()} SAMPLES ---")
    for _, row in sub.iterrows():
        print(f"[{row['category']}] HEADLINE: {row['headline']}")
        print(f"TEXT[:150]: {row['text'][:150]}...\n")
