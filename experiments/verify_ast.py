import json
import ast
import sys

notebook_paths = [
    '/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb',
    '/home/jmomoh/dsn-ai-bootcamp/multilingual_topic_headline_dsn2026.ipynb'
]

overall_success = True

for path in notebook_paths:
    print(f"\n================ Auditing: {path} ================")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")
    path_valid = True

    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            # Filter out bash magic lines like !pip or %load
            py_lines = []
            for line in source.split('\n'):
                if line.strip().startswith('!') or line.strip().startswith('%'):
                    py_lines.append('# ' + line)
                else:
                    py_lines.append(line)
            cleaned_source = '\n'.join(py_lines)
            try:
                ast.parse(cleaned_source)
                print(f"  Cell {idx:02d} ({cleaned_source.splitlines()[0] if cleaned_source.splitlines() else 'empty'}): Syntax OK")
            except SyntaxError as e:
                print(f"  Cell {idx:02d}: SYNTAX ERROR -> {e}")
                path_valid = False
                overall_success = False

    if path_valid:
        print(f"--> ALL CODE CELLS PASSED SYNTAX VALIDATION (100% CLEAN)")
    else:
        print(f"--> SYNTAX ERRORS DETECTED in {path}")

if overall_success:
    print("\n[AST SUCCESS] All notebook files are 100% syntactically valid Python!")
    sys.exit(0)
else:
    print("\n[AST FAILURE] Syntax errors detected.")
    sys.exit(1)
