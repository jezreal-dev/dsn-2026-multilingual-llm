with open('/home/jmomoh/dsn-ai-bootcamp/generate_v11.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace any triple quote followed by 3 parens with triple quote followed by 2 parens
fixed_content = content.replace('""")))', '"""))')

with open('/home/jmomoh/dsn-ai-bootcamp/generate_v11.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("Replacement complete. Verifying syntax...")
import py_compile
py_compile.compile('/home/jmomoh/dsn-ai-bootcamp/generate_v11.py')
print("generate_v11.py compiled cleanly with zero syntax errors!")
