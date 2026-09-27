import json

with open('/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [5, 6, 7, 8, 9, 10, 11]:
    cell = nb['cells'][idx]
    print(f"\n{'='*30} CELL {idx} {'='*30}")
    print("".join(cell.get('source', [])))
