import json

with open('/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb') as f:
    nb = json.load(f)

print("=== CELL 3 SOURCE ===")
print("".join(nb['cells'][3]['source']))
