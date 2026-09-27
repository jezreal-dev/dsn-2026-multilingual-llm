with open('/home/jmomoh/dsn-ai-bootcamp/kaggle_output/notebookdc2758c08b.log') as f:
    lines = f.readlines()

for line in lines:
    if "'loss':" in line or "step" in line.lower() and "epoch" in line.lower():
        print(line.strip())
