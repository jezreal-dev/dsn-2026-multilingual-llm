import json

with open('/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb') as f:
    nb = json.load(f)

print("=== CELL 4 SOURCE ===")
print("".join(nb['cells'][4]['source']))

print("\n=== CELL 6 SOURCE (First 40 lines) ===")
print("".join(nb['cells'][6]['source'][:40]))
