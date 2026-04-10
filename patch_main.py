import re

with open("backend/src/tarok/__main__.py", "r") as f:
    text = f.read()

# Replace use_v2v3=True -> expert_source="v2v3v5"
text = re.sub(
    r"use_v2v3=True",
    r'expert_source="v2v3v5"',
    text
)

# And if there was any use_v2v3=args.use_v2v3 we should fix that too, but let's assume it was just kwargs
with open("backend/src/tarok/__main__.py", "w") as f:
    f.write(text)
print("Done patching __main__.py")
