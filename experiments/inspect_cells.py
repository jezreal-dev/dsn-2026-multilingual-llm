import json
import shutil

# 1. Create a safe backup of Version 6
shutil.copyfile(
    '/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb',
    '/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b_v6_backup.ipynb'
)

# 2. Inspect cells
with open('/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    source_lines = cell.get('source', [])
    first_line = source_lines[0].strip() if source_lines else "<EMPTY>"
    print(f"Cell {idx} ({cell['cell_type']}): {first_line[:90]}")
