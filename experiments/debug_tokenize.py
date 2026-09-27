import tokenize
import io

with open('/home/jmomoh/dsn-ai-bootcamp/generate_v11.py', 'rb') as f:
    tokens = list(tokenize.tokenize(f.readline))

for t in tokens:
    if 220 <= t.start[0] <= 235:
        print(f"Line {t.start[0]}:{t.start[1]} - {tokenize.tok_name[t.exact_type]}: {repr(t.string)}")
