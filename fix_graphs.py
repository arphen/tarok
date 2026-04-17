import re

path = "frontend/src/components/TrainingDashboard.tsx"
with open(path, "r") as f:
    text = f.read()

old_xaxis = '<XAxis dataKey="s" stroke="#666" fontSize={11} />'
new_xaxis = '<XAxis dataKey="s" stroke="#666" fontSize={11} type="number" domain={[1, metrics?.total_sessions || Math.max(sessions, 1)]} />'

text = text.replace(old_xaxis, new_xaxis)

with open(path, "w") as f:
    f.write(text)
