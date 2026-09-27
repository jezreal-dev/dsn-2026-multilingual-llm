import pandas as pd
import numpy as np

train_df = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/data/train.csv')
dev_df = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/data/dev.csv')
test_df = pd.read_csv('/home/jmomoh/dsn-ai-bootcamp/data/test.csv')

print("=== TRAIN DATASET OVERVIEW ===")
print(f"Total Rows: {len(train_df)}")
print("\nTrain Language Counts:")
print(train_df['lang'].value_counts())
print("\nTrain Category Counts:")
print(train_df['category'].value_counts())
print("\nTrain Language x Category Distribution:")
print(pd.crosstab(train_df['lang'], train_df['category'], margins=True))

print("\n=== DEV DATASET OVERVIEW ===")
print(f"Total Rows: {len(dev_df)}")
print("\nDev Language Counts:")
print(dev_df['lang'].value_counts())
print("\nDev Language x Category Distribution:")
print(pd.crosstab(dev_df['lang'], dev_df['category'], margins=True))

print("\n=== TEST DATASET OVERVIEW ===")
print(f"Total Rows: {len(test_df)}")
print(test_df['lang'].value_counts())

train_df['headline_len_words'] = train_df['headline'].fillna('').apply(lambda x: len(x.split()))
print("\n=== TRAIN HEADLINE WORD LENGTH BY LANGUAGE ===")
print(train_df.groupby('lang')['headline_len_words'].describe())

train_df['text_len_words'] = train_df['text'].fillna('').apply(lambda x: len(x.split()))
print("\n=== TRAIN ARTICLE TEXT WORD LENGTH BY LANGUAGE ===")
print(train_df.groupby('lang')['text_len_words'].describe())
