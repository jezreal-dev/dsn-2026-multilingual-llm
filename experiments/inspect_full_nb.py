import json

with open('/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    first_line = cell['source'][0].strip() if cell['source'] else "<EMPTY>"
    print(f"Cell {idx:02d} [{ctype:8s}]: {first_line[:80]}")
