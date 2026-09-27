import json

with open('/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb') as f:
    nb = json.load(f)

for idx in [7, 8, 9, 10]:
    print(f"=== CELL {idx} ===")
    src = "".join(nb['cells'][idx]['source'])
    print(src[:600])
    print("...\n")
