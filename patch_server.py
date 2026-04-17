import re

with open("backend/src/tarok/adapters/api/server.py", "r") as f:
    text = f.read()

old_glob = """    files = sorted(ckpt_dir.glob("tarok_agent_ep*.pt"))"""
new_glob = """    files = [f for f in ckpt_dir.glob("tarok_agent_*.pt") if f.name != "tarok_agent_latest.pt"]
    files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)"""
text = text.replace(old_glob, new_glob)

with open("backend/src/tarok/adapters/api/server.py", "w") as f:
    f.write(text)
