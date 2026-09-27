import zipfile
import glob
import pandas as pd
from pathlib import Path

# Unzip latest leaderboard zip
zip_path = '/home/jmomoh/dsn-ai-bootcamp/leaderboard/dsn-bootcamp-hackathon-2026-llm-agent-track.zip'
out_dir = '/home/jmomoh/dsn-ai-bootcamp/leaderboard/'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(out_dir)

files = sorted(glob.glob('/home/jmomoh/dsn-ai-bootcamp/leaderboard/*.csv'))
latest = files[-1]
df = pd.read_csv(latest)
print("LEADERBOARD FILE:", latest)
print("TOTAL TEAMS:", len(df))

df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
df = df.sort_values('Score', ascending=False).reset_index(drop=True)
df['Rank'] = df.index + 1

my_row = df[df['TeamName'].str.contains('jezreal|momoh', case=False, na=False)]
print("\n=== OUR TEAM STANDING ===")
print(my_row[['Rank', 'TeamName', 'Score']])

print("\n=== TOP 15 TEAMS ===")
print(df[['Rank', 'TeamName', 'Score']].head(15).to_string(index=False))

if len(my_row) > 0:
    idx = my_row.index[0]
    print("\n=== TEAMS AROUND US ===")
    print(df[['Rank', 'TeamName', 'Score']].iloc[max(0, idx-4):min(len(df), idx+5)].to_string(index=False))
