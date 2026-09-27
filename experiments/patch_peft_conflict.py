import ast
import json
from pathlib import Path

TARGET_FILES = [
    Path("/home/jmomoh/dsn-ai-bootcamp/kaggle_kernel/notebookdc2758c08b.ipynb"),
    Path("/home/jmomoh/dsn-ai-bootcamp/multilingual_topic_headline_dsn2026.ipynb"),
]

def patch_cell_1():
    source_nb = TARGET_FILES[0]
    with open(source_nb, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Cell 1 is pip installs
    cell_1_lines = nb_data["cells"][1]["source"]
    new_lines = []
    for line in cell_1_lines:
        if "!pip install" in line:
            # Uninstall outdated torchao to prevent PEFT import conflict, then install peft and metrics
            new_lines.append("!pip uninstall -y -q torchao\n")
            new_lines.append("!pip install -q peft evaluate rouge-score bert-score\n")
        else:
            new_lines.append(line)

    nb_data["cells"][1]["source"] = new_lines

    # Run AST check across all code cells
    for idx, cell in enumerate(nb_data["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_text = "".join(cell["source"])
        clean_code = "\n".join(l for l in code_text.splitlines() if not l.strip().startswith("!"))
        try:
            ast.parse(clean_code)
            print(f"Cell {idx} AST validation: PASSED")
        except SyntaxError as e:
            print(f"Cell {idx} FAILED AST at line {e.lineno}: {e.text}")
            raise e

    for path in TARGET_FILES:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=2)
        print(f"Successfully patched {path}")

if __name__ == "__main__":
    patch_cell_1()
