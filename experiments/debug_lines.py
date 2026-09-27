with open('/home/jmomoh/dsn-ai-bootcamp/generate_v11.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(200, 235):
    print(f"{i+1:3d}: {lines[i]}", end='')
