with open('/home/jmomoh/dsn-ai-bootcamp/generate_v11.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '"""' + ')))'
replacement = '"""' + '))'

count = content.count(target)
print(f"Found {count} occurrences of {repr(target)}")

fixed_content = content.replace(target, replacement)
with open('/home/jmomoh/dsn-ai-bootcamp/generate_v11.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("Saved fixed file. Now running generate_v11.py...")
import subprocess
result = subprocess.run(
    ["/home/jmomoh/dsn-ai-bootcamp/.venv/bin/python3", "/home/jmomoh/dsn-ai-bootcamp/generate_v11.py"],
    capture_output=True,
    text=True
)
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)
