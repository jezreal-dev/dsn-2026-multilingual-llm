import pandas as pd

train_df = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/data/train.csv')
LANGUAGE_CATEGORY_MAP = {
    "hau": ["business", "entertainment", "health", "politics", "religion", "sports", "technology"],
    "ibo": ["business", "entertainment", "health", "politics", "religion", "sports"],
    "pcm": ["business", "entertainment", "health", "politics", "sports"],
    "yor": ["entertainment", "health", "politics", "religion", "sports"]
}

def rebalance_stratified(df, target_per_slice=280):
    balanced_dfs = []
    for lang, valid_cats in LANGUAGE_CATEGORY_MAP.items():
        for cat in valid_cats:
            sub = df[(df['lang'] == lang) & (df['category'] == cat)]
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

balanced_df = rebalance_stratified(train_df, target_per_slice=280)
print(f"Original shape: {train_df.shape}")
print(f"Balanced shape (target=280): {balanced_df.shape}")
print(pd.crosstab(balanced_df['lang'], balanced_df['category'], margins=True))
