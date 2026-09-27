import pandas as pd

dev_df = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/data/dev.csv')
for lang in ['ibo', 'hau', 'yor', 'pcm']:
    print(f"\n=== DEV HEADLINES FOR {lang.upper()} ===")
    sub = dev_df[dev_df['lang'] == lang].head(5)
    for _, row in sub.iterrows():
        print(f"[{row['category']}] {row['headline']}")
