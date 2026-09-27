import pandas as pd

sub = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/kaggle_output/submission.csv')
print("Submission shape:", sub.shape)
print("\nTopic distribution in submission:")
print(sub[sub['id'].str.endswith('_topic')]['prediction'].value_counts())

sub['lang'] = sub['id'].apply(lambda x: x.split('_')[0])
print("\nTopic distribution by language in submission:")
print(pd.crosstab(sub[sub['id'].str.endswith('_topic')]['lang'], sub[sub['id'].str.endswith('_topic')]['prediction']))

print("\nSample Headlines across languages:")
for lang in ['yor', 'ibo', 'hau', 'pcm']:
    print(f"\n--- {lang.upper()} TEST HEADLINES ---")
    h_sub = sub[(sub['id'].str.startswith(lang)) & (sub['id'].str.endswith('_headline'))].head(4)
    for _, row in h_sub.iterrows():
        print(f"[{row['id']}]: {row['prediction']}")
